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
def _load_helper(
    block_ptr,
    first: tl.constexpr,
    second: tl.constexpr,
    pad: tl.constexpr,
):
    if first and second:
        block = tl.load(block_ptr, boundary_check=(0, 1), padding_option=pad)
    elif first:
        block = tl.load(block_ptr, boundary_check=(0,), padding_option=pad)
    elif second:
        block = tl.load(block_ptr, boundary_check=(1,), padding_option=pad)
    else:
        block = tl.load(block_ptr)

    return block


@triton.jit
def _attention_kernel_inner(
    acc,
    l_i,
    m_i,
    q,
    k_block_ptr,
    v_block_ptr,
    start_m,
    actual_seq_lens_k,
    actual_seq_lens_q,
    block_min,
    block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
):
    for start_n in range(block_min, block_max, BLOCK_N):
        k = _load_helper(
            k_block_ptr, PADDED_HEAD, MASK_STEPS and (n_extra_tokens != 0), "zero"
        )
        if PRE_LOAD_V:
            v = _load_helper(
                v_block_ptr, MASK_STEPS and (n_extra_tokens != 0), PADDED_HEAD, "zero"
            )
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS:
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full((BLOCK_M,), actual_seq_lens_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = _load_helper(
                v_block_ptr, MASK_STEPS and (n_extra_tokens != 0), PADDED_HEAD, "zero"
            )
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc += tl.dot(p.to(v_block_ptr.type.element_ty), v)
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))

    return acc, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "PRE_LOAD_V": True},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=8,
        ),
    ],
    key=["IS_CAUSAL", "BLOCK_DMODEL"],
)
@triton.jit
def attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    sm_scale,
    l_ptr,
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
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    MAX_SEQ_LENS_Q: tl.constexpr,
    MAX_SEQ_LENS_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    cu_seq_lens_q_start = 0
    cu_seq_lens_k_start = 0
    seq_lens_q = MAX_SEQ_LENS_Q
    seq_lens_k = MAX_SEQ_LENS_K

    n_blocks = tl.cdiv(seq_lens_k, BLOCK_N)
    if IS_CAUSAL:
        n_blocks_seq_lens = tl.cdiv(
            (start_m + 1) * BLOCK_M + seq_lens_k - seq_lens_q, BLOCK_N
        )
        n_blocks = min(n_blocks, n_blocks_seq_lens)

        if n_blocks <= 0:
            o_offset = (
                off_z * o_stride_z
                + cu_seq_lens_q_start * o_stride_m
                + off_h_q * o_stride_h
            )
            o_block_ptr = tl.make_block_ptr(
                base=o_ptr + o_offset,
                shape=(seq_lens_q, BLOCK_DMODEL),
                strides=(o_stride_m, o_stride_n),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_DMODEL),
                order=(1, 0),
            )

            acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=o_ptr.type.element_ty)
            tl.store(o_block_ptr, acc.to(o_ptr.type.element_ty), boundary_check=(0, 1))
            l_ptrs = (
                l_ptr + off_z * HQ * MAX_SEQ_LENS_Q + off_h_q * MAX_SEQ_LENS_Q + offs_m
            )
            tl.store(l_ptrs, tl.full((BLOCK_M,), value=float("inf"), dtype=tl.float32))

            # TODO: Should dropout and return encoded softmax be handled here too?
            return

    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    n_extra_tokens = 0
    if seq_lens_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seq_lens_k
    elif seq_lens_k % BLOCK_N:
        n_extra_tokens = seq_lens_k % BLOCK_N
    PADDED_HEAD: tl.constexpr = ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL

    q_offset = (
        off_z * q_stride_z + off_h_q * q_stride_h + cu_seq_lens_q_start * q_stride_m
    )
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_offset,
        shape=(seq_lens_q, ACTUAL_BLOCK_DMODEL),
        strides=(q_stride_m, q_stride_k),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    k_offset = (
        off_z * k_stride_z + off_h_k * k_stride_h + cu_seq_lens_k_start * k_stride_n
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_offset,
        shape=(ACTUAL_BLOCK_DMODEL, seq_lens_k),
        strides=(k_stride_k, k_stride_n),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    v_offset = (
        off_z * v_stride_z + off_h_k * v_stride_h + cu_seq_lens_k_start * v_stride_k
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + v_offset,
        shape=(seq_lens_k, ACTUAL_BLOCK_DMODEL),
        strides=(v_stride_k, v_stride_n),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504089
    q = _load_helper(q_block_ptr, True, PADDED_HEAD, "zero")
    q = (q * qk_scale).to(q_block_ptr.type.element_ty)

    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seq_lens_q % BLOCK_M == 0)
    if IS_CAUSAL:
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        masked_blocks = padded_block_k
    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N

    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = _attention_kernel_inner(
            acc,
            l_i,
            m_i,
            q,
            k_block_ptr,
            v_block_ptr,
            start_m,
            seq_lens_k,
            seq_lens_q,
            block_min,
            block_max,
            0,
            0,
            0,
            False,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            offs_m,
            offs_n,
            PRE_LOAD_V,
            False,
            PADDED_HEAD,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    if masked_blocks > 0:
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seq_lens_q - seq_lens_k)
        else:
            offs_n_causal = 0

        k_block_ptr = tl.advance(k_block_ptr, (0, n_full_blocks * BLOCK_N))
        v_block_ptr = tl.advance(v_block_ptr, (n_full_blocks * BLOCK_N, 0))

        acc, l_i, m_i = _attention_kernel_inner(
            acc,
            l_i,
            m_i,
            q,
            k_block_ptr,
            v_block_ptr,
            start_m,
            seq_lens_k,
            seq_lens_q,
            block_min,
            block_max,
            offs_n_causal,
            masked_blocks,
            n_extra_tokens,
            IS_CAUSAL,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            offs_m,
            offs_n,
            PRE_LOAD_V,
            True,
            PADDED_HEAD,
        )

    acc = acc / l_i[:, None]
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seq_lens_q - seq_lens_k
    acc = acc.to(o_ptr.type.element_ty)
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full(
                (BLOCK_DMODEL,), causal_start_idx, dtype=tl.int32
            )
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))

    l_ptrs = l_ptr + off_z * HQ * MAX_SEQ_LENS_Q + off_h_q * MAX_SEQ_LENS_Q + offs_m
    overflow_size = end_m_idx - seq_lens_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
        l_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=l_ptrs_mask)
    else:
        tl.store(l_ptrs, m_i + tl.math.log2(l_i))

    o_offset = (
        off_z * o_stride_z + cu_seq_lens_q_start * o_stride_m + off_h_q * o_stride_h
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + o_offset,
        shape=(seq_lens_q, ACTUAL_BLOCK_DMODEL),
        strides=(o_stride_m, o_stride_n),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    # TODO: Do the boundary check optionally.
    tl.store(o_block_ptr, acc, boundary_check=(0, 1))


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

        output = type(self)._attention_kernel_helper(
            xq,
            keys,
            values,
            sm_scale=1 / math.sqrt(self.head_dim),
            causal=True,
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

    @staticmethod
    def _attention_kernel_helper(q, k, v, sm_scale=1.0, causal=False):
        o = torch.empty_like(q, dtype=v.dtype)

        batch, nheads_q, seq_lens_q, head_size = q.shape
        _, nheads_k, seq_lens_k, _ = k.shape

        padded_d_model = max(1 << (head_size - 1).bit_length(), 16)

        m = torch.empty(
            (batch, nheads_q, seq_lens_q),
            device=q.device,
            dtype=torch.float32,
        )

        def grid(meta):
            return (
                triton.cdiv(seq_lens_q, meta["BLOCK_M"]),
                nheads_q,
                batch,
            )

        attention_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            m,
            o,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *o.stride(),
            HQ=nheads_q,
            HK=nheads_k,
            ACTUAL_BLOCK_DMODEL=head_size,
            MAX_SEQ_LENS_Q=seq_lens_q,
            MAX_SEQ_LENS_K=seq_lens_k,
            IS_CAUSAL=causal,
            BLOCK_DMODEL=padded_d_model,
            BATCH_SIZE=q.shape[0],
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
