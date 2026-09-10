// xattn_score.h - XAttention block-scoring (antidiagonal importance) for llama.cpp integration.
// Pure C++ reference implementation of the XAttention prefill block-selection scoring.
// Mirrors xattn/src/Xattention.py (select_mode="inverse") + utils.py find_blocks_chunked.
//
// This module is deliberately free of ggml dependencies so it can be unit-tested against
// the official numpy/python implementation and reused across backends (CPU / Metal validation
// here, CUDA sparse-FA acceleration remotely).

#ifndef XATTN_SCORE_H
#define XATTN_SCORE_H

#include <cstdint>
#include <vector>

namespace xattn {

// Block-scoring parameters (match XAttention: stride, block_size, threshold).
struct score_params {
    int    stride      = 8;          // antidiagonal importance sampling interval (S)
    int    block_size  = 128;        // token-block granularity (fixed per paper)
    float  threshold   = 0.9f;       // cumulative attention mass to retain (tau)
    bool   causal      = true;
    bool   keep_sink   = true;       // always keep the first (sink) KV block
    bool   keep_recent = true;       // always keep each query's own (recent) block
    bool   softmax     = true;       // block-softmax over the folded scores
};

// Dense row-major result of block selection: one mask per (head) -> [q_block][kv_block] bool.
// For GQA with n_head_q query heads and n_head_kv key/value heads, we score per key/value head
// (one scoring head per KV head) and broadcast to the query heads of that group.
struct block_mask {
    int n_head_q;                     // number of query heads
    int n_head_kv;                    // number of kv (scoring) heads
    int n_q_block;                    // number of query blocks
    int n_kv_block;                   // number of kv blocks
    // rows[kv_head*n_q_block + q_block] is a std::vector<uint8_t> of length n_kv_block.
    std::vector<std::vector<uint8_t> > rows;   // 1 = keep block, 0 = drop

    // Per-(scoring-head, q_block, kv_block) attention mass, mirroring attn_sum from
    // Xattention.py. Exposed for cross-checking / debugging; not required for inference.
    // Layout: [n_head_kv * n_q_block * n_kv_block] row-major over (head, q_block, kv_block).
    std::vector<float> attn_sum;
};

// Convenience: one mask per kv head in a compact form is also exported as
// block_score matrix [n_head_kv * n_q_block] x [n_kv_block] of float importance scores.
// This mirrors attn_sum from Xattention.py (kept for debugging/overrides).

// Compute the XAttention block mask for a full (prefill) sequence.
// Q: [n_head_q * n_seq, dim] row-major, K: [n_head_kv * n_seq, dim] row-major.
// n_seq (q) < n_seq (k) for causal prefill (query attends to itself+past; here we support
// q == k == full sequence length of the current prefill chunk against the KV cache).
//
// NOTE: for causal prefill, query length == key length == total sequence of this ubatch.
// The scoring runs over the whole K (all cached keys) and the whole Q (this prefill chunk).
block_mask score_prefill(
        const float * Q, int n_head_q, int q_len, int dim,
        const float * K, int n_head_kv, int k_len,
        const score_params & p);

} // namespace xattn

#endif // XATTN_SCORE_H
