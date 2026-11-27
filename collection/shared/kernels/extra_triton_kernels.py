"""Tunable kernels and independent PyTorch references for collection."""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d(X, W, Y, H: tl.constexpr, CI: tl.constexpr, CO: tl.constexpr,
           K: tl.constexpr, S: tl.constexpr, OH: tl.constexpr, M: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    b = m // (OH * OH)
    oh = (m // OH) % OH
    ow = m % OH
    acc = tl.zeros((BM, BN), tl.float32)
    for tile in range(tl.cdiv(K * K * CI, BK)):
        r = tile * BK + tl.arange(0, BK)
        ih = oh[:, None] * S + (r // (K * CI))[None, :] - K // 2
        iw = ow[:, None] * S + ((r // CI) % K)[None, :] - K // 2
        x_offset = ((b[:, None] * H + ih) * H + iw) * CI + (r % CI)[None, :]
        x = tl.load(X + x_offset,
                    (m[:, None] < M) & (r[None, :] < K * K * CI)
                    & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < H), 0)
        w = tl.load(W + n[None, :] * (K * K * CI) + r[:, None],
                    (r[:, None] < K * K * CI) & (n[None, :] < CO), 0)
        acc += tl.dot(x, w)
    tl.store(Y + m[:, None] * CO + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < CO))


@triton.jit
def attention(Q, K, V, O, QL: tl.constexpr, KL: tl.constexpr, D: tl.constexpr,
              CAUSAL: tl.constexpr, SCALE: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr):
    head = tl.program_id(1)
    qrow = tl.program_id(0) * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    q = tl.load(Q + (head * QL + qrow[:, None]) * D + d[None, :],
                qrow[:, None] < QL, 0)
    maxima = tl.full((BM,), -float("inf"), tl.float32)
    sums = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for tile in range(tl.cdiv(KL, BN)):
        krow = tile * BN + tl.arange(0, BN)
        offsets = (head * KL + krow[:, None]) * D + d[None, :]
        k = tl.load(K + offsets, krow[:, None] < KL, 0)
        v = tl.load(V + offsets, krow[:, None] < KL, 0)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        mask = krow[None, :] < KL
        if CAUSAL:
            mask = mask & (krow[None, :] <= qrow[:, None])
        scores = tl.where(mask, scores, -float("inf"))
        new_max = tl.maximum(maxima, tl.max(scores, 1))
        # A later causal tile can be entirely masked; new_max remains finite
        # because every query could attend to key zero in the first tile.
        weights = tl.exp(scores - new_max[:, None])
        scale_old = tl.exp(maxima - new_max)
        sums = sums * scale_old + tl.sum(weights, 1)
        acc = acc * scale_old[:, None] + tl.dot(weights.to(v.dtype), v)
        maxima = new_max
    tl.store(O + (head * QL + qrow[:, None]) * D + d[None, :],
             acc / sums[:, None], qrow[:, None] < QL)


@triton.jit
def gather(X, I, O, N: tl.constexpr, W: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    index = tl.load(I + r, r < N, 0)
    mask = (r[:, None] < N) & (c[None, :] < W)
    value = tl.load(X + index[:, None] * W + c[None, :], mask, 0)
    tl.store(O + r[:, None] * W + c[None, :], value, mask)


@triton.jit
def residual_layernorm(X, R, G, B, O, N: tl.constexpr, W: tl.constexpr,
                       BR: tl.constexpr, BC: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, BC)
    offset = r[:, None] * W + c[None, :]
    mask = (r[:, None] < N) & (c[None, :] < W)
    x = tl.load(X + offset, mask, 0).to(tl.float32)
    residual = tl.load(R + offset, mask, 0).to(tl.float32)
    z = x + residual
    mean = tl.sum(z, 1) / W
    centered = tl.where(c[None, :] < W, z - mean[:, None], 0.0)
    inv_std = tl.rsqrt(tl.sum(centered * centered, 1) / W + 1e-5)
    gain = tl.load(G + c, c < W, 0).to(tl.float32)
    bias = tl.load(B + c, c < W, 0).to(tl.float32)
    tl.store(O + offset, centered * inv_std[:, None] * gain[None, :] + bias[None, :], mask)


def make_workload(family, shape, dtype, generator):
    def randn(*dims):
        return torch.randn(dims, generator=generator).to(device="cuda", dtype=dtype)

    if family == "conv2d":
        b, h, ci, co, k, stride = shape
        oh = (h + 2 * (k // 2) - k) // stride + 1
        m = b * oh * oh
        x, w = randn(b, h, h, ci), randn(co, k, k, ci)
        w.div_(math.sqrt(k * k * ci))
        reference = F.conv2d(
            x.permute(0, 3, 1, 2).float(), w.permute(0, 3, 1, 2).float(),
            stride=stride, padding=k // 2,
        ).permute(0, 2, 3, 1).to(dtype).contiguous()
        output = torch.empty_like(reference)

        def launch(bm, bn, bk, warps):
            conv2d[(triton.cdiv(m, bm), triton.cdiv(co, bn))](
                x, w, output, h, ci, co, k, stride, oh, m, bm, bn, bk,
                num_warps=warps, num_stages=2,
            )
    elif family == "attention":
        heads, q_len, kv_len, d, causal = shape
        q, k, v = randn(heads, q_len, d), randn(heads, kv_len, d), randn(heads, kv_len, d)
        scale = d ** -0.5
        logits = (q.float() @ k.float().transpose(-1, -2)) * scale
        if causal:
            mask = torch.ones((q_len, kv_len), device="cuda", dtype=torch.bool).triu(1)
            logits.masked_fill_(mask, -float("inf"))
        reference = (logits.softmax(-1) @ v.float()).to(dtype)
        output = torch.empty_like(reference)

        def launch(bm, bn, warps):
            attention[(triton.cdiv(q_len, bm), heads)](
                q, k, v, output, q_len, kv_len, d, causal, scale, bm, bn,
                num_warps=warps, num_stages=2,
            )
    elif family == "gather":
        vocab, n, width, locality = shape
        x = randn(vocab, width)
        if locality == "uniform":
            index = torch.randint(vocab, (n,), generator=generator)
        elif locality == "clustered":
            clusters = torch.randint(vocab // 128, (triton.cdiv(n, 32),), generator=generator)
            index = clusters.repeat_interleave(32)[:n] * 128
            index += torch.randint(128, (n,), generator=generator)
        else:
            hot = torch.randint(vocab, (16,), generator=generator)
            index = hot[torch.randint(16, (n,), generator=generator)]
        index = index.to(device="cuda", dtype=torch.int32)
        reference = x.index_select(0, index)
        output = torch.empty_like(reference)

        def launch(br, bc, warps):
            gather[(triton.cdiv(n, br), triton.cdiv(width, bc))](
                x, index, output, n, width, br, bc, num_warps=warps,
            )
    elif family == "residual_layernorm":
        n, width = shape
        x, residual, gain, bias = randn(n, width), randn(n, width), randn(width), randn(width)
        reference = F.layer_norm(
            x.float() + residual.float(), (width,), gain.float(), bias.float(),
        ).to(dtype)
        output = torch.empty_like(reference)

        def launch(br, warps):
            residual_layernorm[(triton.cdiv(n, br),)](
                x, residual, gain, bias, output, n, width, br,
                triton.next_power_of_2(width), num_warps=warps,
            )
    else:
        raise ValueError(f"Unknown kernel family: {family}")
    return launch, output, reference
