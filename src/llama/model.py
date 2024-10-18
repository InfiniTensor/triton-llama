# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


@triton.jit
def rms_norm_kernel(x_ptr, y_ptr, w_ptr, stride, n, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)

    y_ptr += row * stride
    x_ptr += row * stride

    rms = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, n, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(x_ptr + cols, mask=cols < n, other=0.0).to(tl.float32)
        rms += a * a
    rms = tl.sqrt(tl.sum(rms) / n + eps)

    for off in range(0, n, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n
        w = tl.load(w_ptr + cols, mask=mask)
        x = tl.load(
            x_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        x_hat = x / rms
        y = x_hat * w
        tl.store(y_ptr + cols, y, mask=mask)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        y = torch.empty_like(x)
        x = x.view(-1, x.shape[-1])
        m, n = x.shape
        stride = x.stride(0)
        w = self.weight
        eps = self.eps
        BLOCK_SIZE = 1024

        rms_norm_kernel[(m,)](x, y, w, stride, n, eps, BLOCK_SIZE=BLOCK_SIZE)

        return y


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.




    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


@triton.jit
def _attention_load_helper(
    block_ptr,
    FIRST: tl.constexpr,
    SECOND: tl.constexpr,
    PADDING: tl.constexpr,
):
    if FIRST and SECOND:
        block = tl.load(block_ptr, boundary_check=(0, 1), padding_option=PADDING)
    elif FIRST:
        block = tl.load(block_ptr, boundary_check=(0,), padding_option=PADDING)
    elif SECOND:
        block = tl.load(block_ptr, boundary_check=(1,), padding_option=PADDING)
    else:
        block = tl.load(block_ptr)

    return block


@triton.jit
def _attention_kernel_inner(
    q,
    k_block_ptr,
    v_block_ptr,
    acc,
    m_i,
    l_i,
    seq_len_q,
    seq_len_k,
    block_min,
    block_max,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PAD_HEAD: tl.constexpr,
    APPLY_MASKING: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    for off_n in range(block_min, block_max, BLOCK_SIZE_N):
        k = _attention_load_helper(k_block_ptr, PAD_HEAD, APPLY_MASKING, "zero")
        qk = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        if APPLY_MASKING and off_n + BLOCK_SIZE_N == block_max:
            boundary = tl.full((BLOCK_SIZE_M,), seq_len_k, dtype=tl.int32)
            mask = off_n + OFFS_N[None, :] < boundary[:, None]
            qk = tl.where(mask, qk, float("-inf"))
        if IS_CAUSAL:
            causal_offs_n = OFFS_N + (seq_len_q - seq_len_k)
            causal_boundary = off_n + causal_offs_n
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        qk += tl.dot(q, k)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.exp2(qk)
        l_ij = tl.sum(p, 1)

        v = _attention_load_helper(v_block_ptr, APPLY_MASKING, PAD_HEAD, "zero")
        alpha = tl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_block_ptr.type.element_ty), v)
        m_i = m_ij
        l_i = l_i * alpha + l_ij

        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_SIZE_N, 0))
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_SIZE_N))

    return acc, m_i, l_i


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32}, num_stages=4, num_warps=8
        ),
    ],
    key=["EMB_DIM", "IS_CAUSAL"],
)
@triton.jit
def attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_stride_z,
    q_stride_h,
    q_stride_m,
    q_stride_k,
    k_stride_z,
    k_stride_h,
    k_stride_n,
    k_stride_k,
    v_stride_z,
    v_stride_h,
    v_stride_k,
    v_stride_n,
    o_stride_z,
    o_stride_h,
    o_stride_m,
    o_stride_n,
    sm_scale,
    ACTUAL_EMB_DIM: tl.constexpr,
    NUM_HEADS_Q: tl.constexpr,
    NUM_HEADS_K: tl.constexpr,
    SEQ_LEN_Q: tl.constexpr,
    SEQ_LEN_K: tl.constexpr,
    EMB_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    off_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)

    offs_m_start = off_m * BLOCK_SIZE_M
    offs_m_end = offs_m_start + BLOCK_SIZE_M
    offs_m = offs_m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    num_blocks = tl.cdiv(SEQ_LEN_K, BLOCK_SIZE_N)

    if IS_CAUSAL:
        num_blocks_adjusted = tl.cdiv(offs_m_end + SEQ_LEN_K - SEQ_LEN_Q, BLOCK_SIZE_N)
        num_blocks = tl.minimum(num_blocks, num_blocks_adjusted)

        if num_blocks <= 0:
            o_off = off_z * o_stride_z + off_h_q * o_stride_h
            o_block_ptr = tl.make_block_ptr(
                base=o_ptr + o_off,
                shape=(SEQ_LEN_Q, EMB_DIM),
                strides=(o_stride_m, o_stride_n),
                offsets=(offs_m_start, 0),
                block_shape=(BLOCK_SIZE_M, EMB_DIM),
                order=(1, 0),
            )
            tl.store(
                o_block_ptr,
                tl.zeros((BLOCK_SIZE_M, EMB_DIM), dtype=o_ptr.type.element_ty),
                boundary_check=(0, 1),
            )

            return

    GROUP_SIZE: tl.constexpr = NUM_HEADS_Q // NUM_HEADS_K
    off_h_k = off_h_q // GROUP_SIZE

    q_off = off_z * q_stride_z + off_h_q * q_stride_h
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_off,
        shape=(SEQ_LEN_Q, ACTUAL_EMB_DIM),
        strides=(q_stride_m, q_stride_k),
        offsets=(offs_m_start, 0),
        block_shape=(BLOCK_SIZE_M, EMB_DIM),
        order=(1, 0),
    )
    k_off = off_z * k_stride_z + off_h_k * k_stride_h
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_off,
        shape=(ACTUAL_EMB_DIM, SEQ_LEN_K),
        strides=(k_stride_k, k_stride_n),
        offsets=(0, 0),
        block_shape=(EMB_DIM, BLOCK_SIZE_N),
        order=(0, 1),
    )
    v_off = off_z * v_stride_z + off_h_k * v_stride_h
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + v_off,
        shape=(SEQ_LEN_K, ACTUAL_EMB_DIM),
        strides=(v_stride_k, v_stride_n),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_N, EMB_DIM),
        order=(1, 0),
    )

    PAD_HEAD: tl.constexpr = ACTUAL_EMB_DIM != EMB_DIM

    q = (
        _attention_load_helper(q_block_ptr, True, PAD_HEAD, "zero")
        * sm_scale
        * 1.44269504089
    ).to(q_block_ptr.type.element_ty)

    num_extra_tokens = 0
    if SEQ_LEN_K < BLOCK_SIZE_N:
        num_extra_tokens = BLOCK_SIZE_N - SEQ_LEN_K
    elif SEQ_LEN_K % BLOCK_SIZE_N:
        num_extra_tokens = SEQ_LEN_K % BLOCK_SIZE_N
    pad_block_k = num_extra_tokens != 0
    if IS_CAUSAL:
        is_modulo_mn = not pad_block_k and (SEQ_LEN_Q % BLOCK_SIZE_M == 0)
        masked_blocks = BLOCK_SIZE_M // BLOCK_SIZE_N + (not is_modulo_mn)
    else:
        masked_blocks = pad_block_k
    masked_blocks = min(masked_blocks, num_blocks)
    num_full_blocks = num_blocks - masked_blocks

    acc = tl.zeros((BLOCK_SIZE_M, EMB_DIM), dtype=tl.float32)
    m_i = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_SIZE_M,), 1, dtype=tl.float32)

    block_min = 0
    if num_full_blocks > 0:
        block_max = (num_blocks - masked_blocks) * BLOCK_SIZE_N

        acc, m_i, l_i = _attention_kernel_inner(
            q,
            k_block_ptr,
            v_block_ptr,
            acc,
            m_i,
            l_i,
            SEQ_LEN_Q,
            SEQ_LEN_K,
            block_min,
            block_max,
            offs_m,
            offs_n,
            PAD_HEAD,
            False,
            False,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )

        block_min = block_max

    block_max = num_blocks * BLOCK_SIZE_N
    if masked_blocks > 0:
        k_block_ptr = tl.advance(k_block_ptr, (0, num_full_blocks * BLOCK_SIZE_N))
        v_block_ptr = tl.advance(v_block_ptr, (num_full_blocks * BLOCK_SIZE_N, 0))

        acc, m_i, l_i = _attention_kernel_inner(
            q,
            k_block_ptr,
            v_block_ptr,
            acc,
            m_i,
            l_i,
            SEQ_LEN_Q,
            SEQ_LEN_K,
            block_min,
            block_max,
            offs_m,
            offs_n,
            PAD_HEAD,
            pad_block_k,
            IS_CAUSAL,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )

    acc /= l_i[:, None]

    if IS_CAUSAL:
        causal_start = SEQ_LEN_Q - SEQ_LEN_K
        if causal_start > offs_m_start and causal_start < offs_m_end:
            mask = offs_m[:, None] >= tl.full((EMB_DIM,), causal_start, dtype=tl.int32)
            acc = tl.where(mask, acc, 0)

    o_off = off_z * o_stride_z + off_h_q * o_stride_h
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + o_off,
        shape=(SEQ_LEN_Q, ACTUAL_EMB_DIM),
        strides=(o_stride_m, o_stride_n),
        offsets=(offs_m_start, 0),
        block_shape=(BLOCK_SIZE_M, EMB_DIM),
        order=(1, 0),
    )
    tl.store(o_block_ptr, acc.to(o_ptr.type.element_ty), boundary_check=(0, 1))


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        output = type(self)._launch_attention_kernel(
            xq,
            keys,
            values,
            is_causal=True,
            sm_scale=1 / math.sqrt(self.head_dim),
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

    @staticmethod
    def _launch_attention_kernel(q, k, v, is_causal=False, sm_scale=1.0):
        o = torch.empty_like(q, dtype=v.dtype)

        batch_size, num_heads_q, seq_len_q, head_size = q.shape
        _, num_heads_k, seq_len_k, _ = k.shape

        padded_head_size = max(1 << (head_size - 1).bit_length(), 16)

        def grid(meta):
            return (
                triton.cdiv(seq_len_q, meta["BLOCK_SIZE_M"]),
                num_heads_q,
                batch_size,
            )

        attention_kernel[grid](
            q,
            k,
            v,
            o,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *o.stride(),
            sm_scale=sm_scale,
            ACTUAL_EMB_DIM=head_size,
            NUM_HEADS_Q=num_heads_q,
            NUM_HEADS_K=num_heads_k,
            SEQ_LEN_Q=seq_len_q,
            SEQ_LEN_K=seq_len_k,
            EMB_DIM=padded_head_size,
            IS_CAUSAL=is_causal,
        )

        return o


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len * 2,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
