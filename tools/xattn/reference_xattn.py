#!/usr/bin/env python3
"""Exact numpy port of the official XAttention prefill block-selection pipeline.

Single-chunk faithful reproduction of xattn/src/Xattention.py (select_mode="inverse",
non-triton) + utils.py find_blocks_chunked (threshold+causal) + Xattention.py:275-297
post-processing. One chunk covers all query blocks (q_block_num), which is the
single-shot prefill case we integrate into llama.cpp.

Validates the C++ scorer (llama.cpp/src/xattn-score.cpp via tools/xattn/test-xattn).

Usage: python3 reference_xattn.py <outfile> [seed q_len k_len dim n_head stride block thr]
"""
import sys
import numpy as np

_rng_state = 0x9e3779b97f4a7c15

def rng_next():
    global _rng_state
    x = _rng_state
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 7
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    _rng_state = x
    return x

def frand(lo, hi):
    v = rng_next()
    u = (v >> 40) / (1 << 24)
    return lo + (hi - lo) * u

# folded antidiagonal scores (inverse mode): S[h,i,j] = sum_s sum_d Q[h,i*S+(S-1-s),d]*K[h,j*S+s,d]
def antidiag_scores(Q, K, stride):
    nh, q_len, dim = Q.shape
    _, k_len, _ = K.shape
    QF = (q_len + stride - 1) // stride
    KF = (k_len + stride - 1) // stride
    S = np.zeros((nh, QF, KF), np.float32)
    for h in range(nh):
        for i in range(QF):
            for j in range(KF):
                acc = 0.0
                for s in range(stride):
                    qi = i * stride + (stride - 1 - s)
                    kj = j * stride + s
                    if qi >= q_len or kj >= k_len:
                        continue
                    acc += float(np.dot(Q[h, qi], K[h, kj]))
                S[h, i, j] = acc
    return S

# folded -> block softmax -> folded-space block attention mass (faithful to Xattention.py)
# W[i][j] = softmax_j(scores[i][j]); block qb = i//rbs, kv block kb2 = j//rbs, rbs = block/stride
def block_attn_sum(scores, q_len, k_len, block_size, stride):
    nh, QF, KF = scores.shape
    n_q_blk = (q_len + block_size - 1) // block_size
    n_kv_blk = (k_len + block_size - 1) // block_size
    rbs = block_size // stride
    if rbs < 1:
        return np.zeros((nh, n_q_blk, n_kv_blk), np.float32)
    A = np.zeros((nh, n_q_blk, n_kv_blk), np.float32)
    for h in range(nh):
        mx = scores[h].max(axis=1, keepdims=True)
        e = np.exp(scores[h] - mx)
        W = e / e.sum(axis=1, keepdims=True)
        for i in range(QF):
            qb = i // rbs
            if qb >= n_q_blk:
                continue
            for j in range(KF):
                kb = j // rbs
                if kb >= n_kv_blk:
                    continue
                A[h, qb, kb] += W[i, j]
    return A

# EXACT single-chunk port of utils.py find_blocks_chunked (threshold, causal).
# attn_sum: [H, n_qblk, n_kvblk]; chunk_num = n_qblk; current_index = n_kvblk - n_qblk.
def find_blocks_chunked_exact(attn_sum, threshold):
    H, chunk_num, block_num = attn_sum.shape
    total = attn_sum.sum(axis=-1, keepdims=True)
    required = total * threshold
    current_index = block_num - chunk_num

    # forced: sink (block 0) + recent (kv block current_index + r for local query r)
    forced = np.zeros_like(attn_sum, dtype=bool)
    forced[:, :, 0] = True
    for r in range(chunk_num):
        kpos = current_index + r
        if 0 <= kpos < block_num:
            forced[:, r, kpos] = True

    # other = zero the forced scores; sort descending
    other = np.where(forced, 0.0, attn_sum)
    sorted_values = -np.sort(-other, axis=-1)
    forced_sum = np.where(forced, attn_sum, 0.0).sum(axis=-1, keepdims=True)
    merged = np.concatenate([
        np.zeros((H, chunk_num, 1), np.float32),
        forced_sum,
        sorted_values[:, :, :-2],
    ], axis=-1)
    # index ordered by 100000*(1+attn) for forced else attn, descending
    forced_scores = np.where(forced, 100000.0 * (1.0 + attn_sum), attn_sum.astype(np.float64))
    index = np.argsort(-forced_scores, axis=-1, kind="stable")
    merge2 = np.concatenate([np.zeros((H, chunk_num, 1), np.float32), merged[:, :, :-1]], axis=-1)
    cumulative = np.cumsum(merge2, axis=-1)
    index_mask = cumulative < required
    index = np.where(index_mask, index, 0)
    out = np.zeros((H * chunk_num, block_num), bool)
    rows = np.arange(H * chunk_num)[:, None]
    out[rows, index.reshape(H * chunk_num, block_num)] = True
    mask = out.reshape(H, chunk_num, block_num)
    # causal future cut
    for r in range(chunk_num):
        kpos = current_index + r
        if kpos < block_num - 1:
            mask[:, r, kpos + 1:] = False
    return mask

def official_pipeline(Q, K, stride, block_size, threshold):
    nh, q_len, dim = Q.shape
    _, k_len, _ = K.shape
    S = antidiag_scores(Q, K, stride)
    n_q_blk = (q_len + block_size - 1) // block_size
    n_kv_blk = (k_len + block_size - 1) // block_size
    A = block_attn_sum(S, q_len, k_len, block_size, stride)
    mask = find_blocks_chunked_exact(A, threshold)          # [H, n_qblk, n_kvblk]
    qb = n_q_blk
    # post: causal tril on the bottom-right qb x qb submatrix
    tril = np.tril(np.ones((qb, qb), dtype=bool), 0)[None, :, :]
    mask[:, -qb:, -qb:] = np.where(tril, mask[:, -qb:, -qb:], False)
    # post: keep_sink
    mask[:, 0, :] = True
    # post: keep_recent (eye on bottom-right)
    eye = np.eye(qb, dtype=bool)[None, :, :]
    mask[:, -qb:, -qb:] = np.where(eye, True, mask[:, -qb:, -qb:])
    return mask

def main():
    args = sys.argv[1:]
    outfile = args[0] if args else "/tmp/xattn_out.txt"
    seed = int(args[1]) if len(args) > 1 else 12345
    q_len = int(args[2]) if len(args) > 2 else 512
    k_len = int(args[3]) if len(args) > 3 else 512
    dim   = int(args[4]) if len(args) > 4 else 64
    n_head= int(args[5]) if len(args) > 5 else 4
    stride= int(args[6]) if len(args) > 6 else 8
    blk   = int(args[7]) if len(args) > 7 else 128
    thr   = float(args[8]) if len(args) > 8 else 0.9

    global _rng_state
    _rng_state = seed
    Q = np.zeros((n_head, q_len, dim), np.float32)
    K = np.zeros((n_head, k_len, dim), np.float32)
    for h in range(n_head):
        for t in range(q_len):
            for d in range(dim): Q[h, t, d] = frand(-1, 1)
    for h in range(n_head):
        for t in range(k_len):
            for d in range(dim): K[h, t, d] = frand(-1, 1)

    M = official_pipeline(Q, K, stride, blk, thr).astype(np.int32)

    with open(outfile) as f:
        toks = f.read().split()
    hh, nqb, nkb, total, cs = map(int, toks[:5])
    flat = np.array(list(map(int, toks[5:5 + total])), np.int32).reshape(hh, nqb, nkb)
    diff = int((M != flat).sum())
    print(f"numpy(official) vs cpp: head={n_head} qblk={nqb} kblk={nkb}")
    print(f"  numpy kept={int(M.sum())}  cpp kept={int(flat.sum())}")
    print(f"  mismatched={diff}/{M.size} ({100*diff/M.size:.2f}%)")
    if diff == 0:
        print("  MATCH OK")
    else:
        idx = np.argwhere(M != flat)
        for h_, qb, kb in idx[:12]:
            print(f"  h={h_} qblk={qb} kblk={kb}: np={int(M[h_,qb,kb])} cpp={int(flat[h_,qb,kb])}")

if __name__ == "__main__":
    main()
