# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
import math
import os
from typing import Optional

import einops
import torch
from packaging import version
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.transformer import shard_heads
from anemoi.models.distributed.transformer import shard_sequence
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class MultiHeadSelfAttention(nn.Module):
    """Multi Head Self Attention Pytorch Layer

    allows for three different attention implementations:
    - scaled dot product attention, see https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    - flash attention, see https://github.com/Dao-AILab/flash-attention
    - flex attention, see https://pytorch.org/blog/flexattention/
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        layer_kernels: DotDict,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        is_causal: bool = False,
        window_size: Optional[int] = None,
        dropout_p: float = 0.0,
        attention_implementation: str = "flash_attention",
        softcap: Optional[float] = None,
        use_alibi_slopes: bool = False,
    ):
        """Initialize MultiHeadSelfAttention.

        For the flash attention implementation, two additional parameters are available: softcap, use_alibi_slopes

        softcap: Softcapping prevents the logits from growing excessively large

        use_alibi_slopes: Adds bias of `(-alibi_slope * |i + seqlen_k - seqlen_q - j|)` to the attention score of
        query i and key j, where alibi_slope is calculated using get_alibi_slopes

        Parameters
        ----------
        num_heads : int
            number of heads
        embed_dim : int
            embedding dimension
        qkv_bias : bool, optional
            bias for querys, keys and values, by default False
        qk_norm : bool, optional
            normalize q and k, by default False
        is_causal : bool, optional
            apply causal attention mask, by default False
        window_size : Optional[int], optional
            window_size, by default None
        dropout_p : float, optional
            dropout probability, by default 0.0
        attention_implementation: str, optional
            A predefined string which selects which underlying attention
            implementation, by default "flash_attention"
        softcap : float, optional
            Anything > 0 activates softcapping attention, by default None
        use_alibi_slopes : bool, optional
            Adds bias
        """
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"

        self.attention_implementation = attention_implementation
        self.use_alibi_slopes = use_alibi_slopes

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads  # q k v
        self.window_size = window_size
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.qk_norm = qk_norm
        self.softcap = softcap

        self.set_attention_function()

        if self.use_alibi_slopes:
            self.alibi_slopes = get_alibi_slopes(num_heads)
            assert self.alibi_slopes.shape[0] == num_heads, "Error: Number of alibi_slopes must match number of heads"
        else:
            self.alibi_slopes = None

        linear = layer_kernels["Linear"]
        self.lin_qkv = linear(embed_dim, 3 * embed_dim, bias=qkv_bias)

        self.projection = linear(embed_dim, embed_dim, bias=True)

        if self.qk_norm:
            self.q_norm = layer_kernels["QueryNorm"](self.head_dim)
            self.k_norm = layer_kernels["KeyNorm"](self.head_dim)

    def set_attention_function(self):
        attn_funcs = {
            "flash_attention": FlashAttentionWrapper,
            "scaled_dot_product_attention": SDPAAttentionWrapper,
            "flex_attention": FlexAttentionWrapper,
        }
        assert (
            self.attention_implementation in attn_funcs
        ), f"{self.attention_implementation} not supported. \
              Please change model.processor.attention_implementation to one of: {attn_funcs.keys()}"
        LOGGER.info(f"Using {self.attention_implementation}")

        # initalise the attn func here
        self.attention = attn_funcs[self.attention_implementation]()

    def forward(
        self, x: Tensor, shapes: list, batch_size: int, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:

        query, key, value = self.lin_qkv(x).chunk(3, -1)

        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded accross GPUs"

        query, key, value = (
            einops.rearrange(
                t,
                "(batch grid) (heads vars) -> batch heads grid vars",
                batch=batch_size,
                heads=self.num_heads,
            )
            for t in (query, key, value)
        )

        query = shard_heads(query, shapes=shapes, mgroup=model_comm_group)
        key = shard_heads(key, shapes=shapes, mgroup=model_comm_group)
        value = shard_heads(value, shapes=shapes, mgroup=model_comm_group)
        dropout_p = self.dropout_p if self.training else 0.0

        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        out = self.attention(
            query,
            key,
            value,
            batch_size,
            causal=False,
            window_size=self.window_size,
            dropout_p=dropout_p,
            softcap=self.softcap,
            alibi_slopes=self.alibi_slopes,
        )

        out = shard_sequence(out, shapes=shapes, mgroup=model_comm_group)
        out = einops.rearrange(out, "batch heads grid vars -> (batch grid) (heads vars)")

        out = self.projection(out)

        return out


class SDPAAttentionWrapper(nn.Module):
    """Wrapper for Pytorch scaled dot product attention"""

    def __init__(self):
        super().__init__()

        from torch.nn.functional import scaled_dot_product_attention

        self.attention = scaled_dot_product_attention
        self.mask = None
        self.window_size = None

    def update_mask(self, seq_len, window_size: int, device: str):

        self.mask = (
            torch.abs(
                torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
            )
            <= window_size
        )

    def forward(
        self,
        query,
        key,
        value,
        batch_size: int,
        causal=False,
        window_size=None,
        dropout_p=0.0,
        softcap=None,
        alibi_slopes=None,
    ):
        if softcap is not None and softcap > 0:
            raise NotImplementedError(
                "Softcap not supported by Pytorchs SDPA. please switch to flash attention or disable softcap."
            )
        if alibi_slopes is not None:
            raise NotImplementedError(
                "Alibi slopes not supported by Pytorchs SDPA. please switch to flash attention or disable alibi slopes."
            )

        sequence_len = query.shape[-2]

        if window_size is not None and (self.mask is None or tuple(self.mask.shape) != (sequence_len, sequence_len)):
            self.update_mask(sequence_len, window_size=window_size, device=query.device)

        # Let PyTorch choose the best available SDPA backend first.
        # Fallback to MATH only if backend selection fails for this input.
        try:
            out = self.attention(
                query,
                key,
                value,
                attn_mask=self.mask,
                is_causal=causal,
                dropout_p=dropout_p,
            )
        except RuntimeError:
            with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
                out = self.attention(
                    query,
                    key,
                    value,
                    attn_mask=self.mask,
                    is_causal=causal,
                    dropout_p=dropout_p,
                )

        return out


class FlexAttentionWrapper(nn.Module):
    """Wrapper for Pytorch flex attention."""

    def __init__(self):
        super().__init__()
        self._init_attention_ops()
        self.block_mask = None
        self.mask_signature = None

    def _init_attention_ops(self):
        try:
            from torch.nn.attention.flex_attention import create_block_mask
            from torch.nn.attention.flex_attention import flex_attention
        except ImportError as exc:
            raise ImportError(
                "Error: Flex attention is not available in this PyTorch installation. "
                "Please use PyTorch with torch.nn.attention.flex_attention support."
            ) from exc

        self.eager_attention = flex_attention
        self.attention = self.eager_attention
        self._compile_fallback_done = False
        # Only compile flex_attention if model-level compilation is NOT happening.
        # When the model is compiled as a whole unit, individual attention layers should not
        # also compile separately to avoid double-compilation overhead.
        disable_attention_compile = os.environ.get("DISABLE_ATTENTION_COMPILE")
        if hasattr(torch, "compile") and not disable_attention_compile:
            compile_options = {"device": "CPU"}
            try:
                self.attention = torch.compile(
                    flex_attention,
                    backend="openvino",
                    options=compile_options,
                    dynamic=False,
                )
            except Exception as exc:  # pragma: no cover - compile availability is environment-dependent.
                LOGGER.warning(
                    "Unable to compile flex attention with OpenVINO backend. "
                    f"Trying default torch.compile. Error: {exc}"
                )
                try:
                    self.attention = torch.compile(flex_attention, dynamic=False)
                except Exception as exc_default:  # pragma: no cover - compile availability is environment-dependent.
                    LOGGER.warning(
                        "Unable to compile flex attention with default backend. "
                        f"Falling back to eager mode. Error: {exc_default}"
                    )
        self.create_block_mask = create_block_mask

    def __getstate__(self):
        state = self.__dict__.copy()
        # Runtime callables created/imported here are not reliably picklable.
        state["attention"] = None
        state["eager_attention"] = None
        state["create_block_mask"] = None
        state["block_mask"] = None
        state["mask_signature"] = None
        state["_compile_fallback_done"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_attention_ops()
        self.block_mask = None
        self.mask_signature = None

    @staticmethod
    def _build_mask_mod(causal: bool, window_size: Optional[int]):
        if window_size is None and not causal:
            return None

        if causal and window_size is None:

            def mask_mod(batch, head, q_idx, kv_idx):
                return q_idx >= kv_idx

            return mask_mod

        if causal:

            def mask_mod(batch, head, q_idx, kv_idx):
                return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= window_size)

            return mask_mod

        def mask_mod(batch, head, q_idx, kv_idx):
            return torch.abs(q_idx - kv_idx) <= window_size

        return mask_mod

    def _update_block_mask(self, query: Tensor, key: Tensor, causal: bool, window_size: Optional[int]):
        q_len = query.shape[-2]
        kv_len = key.shape[-2]
        batch_size = query.shape[0]
        num_heads = query.shape[1]
        signature = (q_len, kv_len, batch_size, num_heads, str(query.device), causal, window_size)

        if self.mask_signature == signature:
            return

        mask_mod = self._build_mask_mod(causal=causal, window_size=window_size)
        if mask_mod is None:
            self.block_mask = None
        else:
            self.block_mask = self.create_block_mask(
                mask_mod,
                B=batch_size,
                H=num_heads,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                device=query.device,
            )
        self.mask_signature = signature

    def forward(
        self,
        query,
        key,
        value,
        batch_size: int,
        causal: bool = False,
        window_size: int = None,
        dropout_p: float = 0.0,
        softcap: Optional[float] = None,
        alibi_slopes: torch.Tensor = None,
    ):
        if dropout_p > 0.0:
            LOGGER.warning("Dropout is not currently supported by flex attention wrapper and will be ignored.")
        if softcap is not None and softcap > 0:
            raise NotImplementedError(
                "Softcap is not supported by Pytorch flex_attention in this wrapper. "
                "Please disable softcap or switch attention implementation."
            )
        if alibi_slopes is not None:
            raise NotImplementedError(
                "Alibi slopes are not supported by Pytorch flex_attention in this wrapper. "
                "Please disable alibi slopes or switch attention implementation."
            )

        self._update_block_mask(query=query, key=key, causal=causal, window_size=window_size)

        # OpenVINO/TorchDynamo can fail shape-guard creation on non-contiguous strided views.
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        try:
            out = self.attention(
                query,
                key,
                value,
                block_mask=self.block_mask,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime compiler support.
            if self.attention is not self.eager_attention and not self._compile_fallback_done:
                LOGGER.warning(
                    f"Compiled flex attention failed at runtime, falling back to eager mode. Error: {exc}"
                )
                self.attention = self.eager_attention
                self._compile_fallback_done = True
                out = self.attention(
                    query,
                    key,
                    value,
                    block_mask=self.block_mask,
                )
            else:
                raise
        return out


class FlashAttentionWrapper(nn.Module):
    """Wrapper for Flash attention."""

    def __init__(self):
        super().__init__()
        try:
            import flash_attn
        except ImportError:
            raise ImportError("Error: Flash-attn not installed. Please install flash-attn to use Flash Attention")

        if version.parse(flash_attn.__version__) < version.parse("2.6.0"):
            raise RuntimeError("Error: Flash-attn version is too low. Update to 2.6.0 or higher.")
        else:
            self.attention = flash_attn.flash_attn_func

    def forward(
        self,
        query,
        key,
        value,
        batch_size: int,
        causal: bool = False,
        window_size: int = None,
        dropout_p: float = 0.0,
        softcap: Optional[float] = None,
        alibi_slopes: torch.Tensor = None,
    ):
        query, key, value = (
            einops.rearrange(t, "batch heads grid vars -> batch grid heads vars") for t in (query, key, value)
        )

        alibi_slopes = alibi_slopes.repeat(batch_size, 1).to(query.device) if alibi_slopes is not None else None

        out = self.attention(
            query,
            key,
            value,
            causal=False,
            window_size=(window_size, window_size),
            dropout_p=dropout_p,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
        )
        out = einops.rearrange(out, "batch grid heads vars -> batch heads grid vars")
        return out


def get_alibi_slopes(num_heads: int) -> Tensor:
    """Calculates linearly decreasing slopes for alibi attention.

    Parameters
    ----------
    num_heads : int
        number of attention heads

    Returns
    -------
    Tensor
        aLiBi slopes
    """
    n = 2 ** math.floor(math.log2(num_heads))
    slope_0 = 2 ** (-8 / n)
    alibi_slopes = torch.pow(slope_0, torch.arange(1, 1 + n))
    if n < num_heads:
        slope_hat_0 = 2 ** (-4 / n)
        alibi_slopes_hat = torch.pow(slope_hat_0, torch.arange(1, 1 + 2 * (num_heads - n), 2))
        alibi_slopes = torch.cat([alibi_slopes, alibi_slopes_hat])
    return alibi_slopes
