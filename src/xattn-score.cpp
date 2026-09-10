// xattn-score.cpp - XAttention antidiagonal block-scoring (prefill).
// Pure C++ reference of XAttention.py (select_mode="inverse") + find_blocks_chunked.
// One score mask per KV head, broadcast to its GQA query-head group (MSA convention).

#include "xattn-score.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace xattn {

// Fold + antidiagonal sum: build the (folded) QK^T importance matrix for one head.
// Mirrors reshaped_query/reshaped_key + matmul in Xattention.py (non-triton branch).
// Output: scores[qs] where qs indexes query *folded positions* -> q_len/stride rows,
//         ks indexes key folded positions -> k_len/stride cols.
// For inverse mode: score[i,j] = sum_s sum_d Q[(i*S + (S-1-s))%q_len, d] * K[(j*S+s)%k_len, d]
// with masking handled separately (usable positions only).
//
// To keep this faithful yet simple and fast for moderate sizes we compute it directly.
static void antidiag_scores(
        const float * Q, int q_len,
        const float * K, int k_len,
        int dim, int stride,
        std::vector<float> & out) {
    const int QF = (q_len + stride - 1) / stride;   // folded query rows
    const int KF = (k_len + stride - 1) / stride;   // folded key cols
    out.assign((size_t) QF * KF, 0.0f);

    for (int i = 0; i < QF; ++i) {
        for (int j = 0; j < KF; ++j) {
            float acc = 0.0f;
            for (int s = 0; s < stride; ++s) {
                const int qi = i*stride + (stride - 1 - s);
                const int kj = j*stride + s;
                if (qi >= q_len || kj >= k_len) continue;
                const float * qrow = Q + (size_t) qi*dim;
                const float * krow = K + (size_t) kj*dim;
                for (int d = 0; d < dim; ++d) {
                    acc += qrow[d] * krow[d];
                }
            }
            out[(size_t) i*KF + j] = acc;
        }
    }
}

// Folded-space block attention mass, faithful to Xattention.py.
// Mirrors Xattention.py (non-triton) attn_weights_slice + `view/.../sum`:
//   W[i][j] = softmax_j(scores[i][j] / (sqrt(dim)*stride*norm))   [i over folded Q rows, j over folded K cols]
//   attn_sum[qb][kb2] = sum_{i in [qb*rbs,(qb+1)*rbs)} sum_{j in [kb2*rbs,(kb2+1)*rbs)} W[i][j]
// where rbs = block_size / stride (reshaped_block_size). Since rbs*stride == block_size,
// the folded-block index equals the physical block index: qb indexes query blocks and
// kb2 indexes kv blocks directly. The 1/sqrt(dim)/stride scale is row-constant and cancels
// under the per-row softmax, so it is omitted.
// scores: [QF, KF]; result: [n_q_blk, n_kv_blk].
static void block_attn_sum(
        const std::vector<float> & scores, int QF, int KF,
        int q_len, int k_len, int block_size, int stride, bool softmax,
        int n_q_blk, int n_kv_blk,
        std::vector<float> & attn_sum) {
    const int rbs = block_size / stride;               // reshaped_block_size
    if (rbs < 1) {                                    // guard against stride > block_size
        attn_sum.assign((size_t) n_q_blk * n_kv_blk, 0.0f);
        return;
    }

    attn_sum.assign((size_t) n_q_blk * n_kv_blk, 0.0f);

    std::vector<float> fmax(QF, -INFINITY), fsum(QF, 0.0f);
    for (int i = 0; i < QF; ++i) {
        float m = -INFINITY;
        for (int j = 0; j < KF; ++j) {
            float v = scores[(size_t) i*KF + j];
            if (v > m) m = v;
        }
        fmax[i] = m;
        float s = 0.0f;
        for (int j = 0; j < KF; ++j) {
            s += std::exp(scores[(size_t) i*KF + j] - m);
        }
        fsum[i] = s;
    }

    for (int i = 0; i < QF; ++i) {
        const float inv = softmax ? (1.0f / fsum[i]) : 1.0f;
        const int qb = i / rbs;                        // folded-block == physical query block
        if (qb >= n_q_blk) continue;
        for (int j = 0; j < KF; ++j) {
            const float w = std::exp(scores[(size_t) i*KF + j] - fmax[i]) * inv;
            const int kb = j / rbs;                    // folded-block == physical kv block
            if (kb >= n_kv_blk) continue;
            attn_sum[(size_t) qb*n_kv_blk + kb] += w;
        }
    }
}

// find_blocks_chunked replica (single-chunk prefill, threshold mode, causal).
// Exact port of xattn/src/utils.py find_blocks_chunked + Xattention.py:275-297 post-processing.
// One chunk covers all query blocks: chunk_num = n_q_blk, current_index = n_kv_blk - n_q_blk.
// Forced blocks = sink (kv block 0) + recent (kv block current_index + qb for query block qb).
static void select_blocks(
        const std::vector<float> & attn_sum, int n_q_blk, int n_kv_blk,
        const score_params & p,
        std::vector<uint8_t> & mask) {
    const int H = 1;   // this routine handles a single (head, all-q-blocks) plane; caller loops heads
    const int chunk_num = n_q_blk;
    const int block_num = n_kv_blk;
    const int current_index = block_num - chunk_num;

    mask.assign((size_t) n_q_blk * n_kv_blk, 0);

    // required = total * threshold per query block
    std::vector<float> required(n_q_blk);
    for (int qb = 0; qb < n_q_blk; ++qb) {
        float total = 0.0f;
        for (int kb = 0; kb < n_kv_blk; ++kb) total += attn_sum[(size_t) qb*n_kv_blk + kb];
        required[qb] = (total <= 0.0f ? 1.0f : total) * p.threshold;
    }

    // forced mask per (qb, kb): sink + recent
    std::vector<uint8_t> forced((size_t) n_q_blk * n_kv_blk, 0);
    for (int qb = 0; qb < n_q_blk; ++qb) {
        if (p.keep_sink) forced[(size_t) qb*n_kv_blk + 0] = 1;
        if (p.keep_recent) {
            const int kpos = current_index + qb;
            if (kpos >= 0 && kpos < n_kv_blk) forced[(size_t) qb*n_kv_blk + kpos] = 1;
        }
    }

    // other = zero forced per row; handled inside the qb loop below.
    for (int qb = 0; qb < n_q_blk; ++qb) {
        // per-row other + descending order
        std::vector<float> other(n_kv_blk);
        for (int kb = 0; kb < n_kv_blk; ++kb)
            other[kb] = forced[(size_t) qb*n_kv_blk + kb] ? 0.0f : attn_sum[(size_t) qb*n_kv_blk + kb];
        std::vector<int> ord(n_kv_blk);
        for (int i = 0; i < n_kv_blk; ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](int a, int b) {
            if (other[a] != other[b]) return other[a] > other[b];
            return a < b;
        });
        std::vector<float> sorted_val(n_kv_blk);
        for (int i = 0; i < n_kv_blk; ++i) sorted_val[i] = other[ord[i]];

        // forced_sum
        float forced_sum = 0.0f;
        for (int kb = 0; kb < n_kv_blk; ++kb)
            if (forced[(size_t) qb*n_kv_blk + kb]) forced_sum += attn_sum[(size_t) qb*n_kv_blk + kb];

        // merged = [0, forced_sum, sorted_val[0..n-2]]
        std::vector<float> merged;
        merged.push_back(0.0f);
        merged.push_back(forced_sum);
        for (int i = 0; i < n_kv_blk - 2; ++i) merged.push_back(sorted_val[i]);
        // merge2 = [0, merged[0..n-2]] then cumsum (faithful to numpy: cumulative[i] = sum merged[0..i-1])
        std::vector<float> cum(n_kv_blk);
        float acc = 0.0f;
        for (int i = 0; i < n_kv_blk; ++i) {
            cum[i] = acc;                                   // cumulative up to (not including) merged[i]
            if (i < n_kv_blk - 1) acc += merged[i];          // advance (merge2 drops merged[n-1])
        }
        // index: sort by 100000*(1+attn) if forced else attn, desc
        std::vector<int> idx(n_kv_blk);
        for (int i = 0; i < n_kv_blk; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            const double fa = forced[(size_t) qb*n_kv_blk + a] ?
                100000.0 * (1.0 + attn_sum[(size_t) qb*n_kv_blk + a]) :
                (double) attn_sum[(size_t) qb*n_kv_blk + a];
            const double fb = forced[(size_t) qb*n_kv_blk + b] ?
                100000.0 * (1.0 + attn_sum[(size_t) qb*n_kv_blk + b]) :
                (double) attn_sum[(size_t) qb*n_kv_blk + b];
            if (fa != fb) return fa > fb;
            return a < b;
        });
        // index_mask = cum < required; scatter mask[qb, index]=True for ALL positions,
        // with non-satisfying positions zeroed (faithful to `index = where(index_mask, index, 0)`
        // then `mask[..., index] = True` in utils.py -- the zeroed entries force block 0 True).
        for (int i = 0; i < n_kv_blk; ++i) {
            int kb = (cum[i] < required[qb]) ? idx[i] : 0;
            if (kb < n_kv_blk) mask[(size_t) qb*n_kv_blk + kb] = 1;
        }
    } // qb

    // causal future cut (utils.py:176-180): for local query r, kv > current_index + r is dropped
    if (p.causal) {
        for (int qb = 0; qb < n_q_blk; ++qb) {
            const int kpos = current_index + qb;
            for (int kb = kpos + 1; kb < n_kv_blk; ++kb) {
                if (kb > 0 && qb < n_kv_blk) mask[(size_t) qb*n_kv_blk + kb] = 0;
            }
        }
    }

    // post-processing (Xattention.py:275-297) if this caller is the single full-head mask
    const int qb = n_q_blk;
    if (qb <= n_kv_blk) {
        // tril on bottom-right qb x qb submatrix
        for (int r = 0; r < qb; ++r) {
            for (int c = 0; c < qb; ++c) {
                const int kb = (n_kv_blk - qb) + c;
                if (kb < 0 || kb >= n_kv_blk) continue;
                const bool keep = (c <= r);
                const int rr = (n_q_blk - qb) + r;
                if (rr < 0 || rr >= n_q_blk) continue;
                if (!keep) mask[(size_t) rr*n_kv_blk + kb] = 0;
            }
        }
        if (p.keep_sink) {
            for (int kb = 0; kb < n_kv_blk; ++kb) mask[(size_t) 0*n_kv_blk + kb] = 1;
        }
        if (p.keep_recent) {
            for (int r = 0; r < qb; ++r) {
                const int rr = (n_q_blk - qb) + r;
                const int kb = (n_kv_blk - qb) + r;
                if (rr >= 0 && rr < n_q_blk && kb >= 0 && kb < n_kv_blk) {
                    mask[(size_t) rr*n_kv_blk + kb] = 1;
                }
            }
        }
    }
}

block_mask score_prefill(
        const float * Q, int n_head_q, int q_len, int dim,
        const float * K, int n_head_kv, int k_len,
        const score_params & p) {
    block_mask res;
    res.n_head_q   = n_head_q;
    res.n_head_kv  = n_head_kv;
    const int n_q_blk  = std::max(1, (q_len + p.block_size - 1) / p.block_size);
    const int n_kv_blk = std::max(1, (k_len + p.block_size - 1) / p.block_size);
    res.n_q_block  = n_q_blk;
    res.n_kv_block = n_kv_blk;

    // one mask per KV (scoring) head; broadcast to the group
    res.rows.assign((size_t) n_head_kv * n_q_blk, std::vector<uint8_t>(n_kv_blk));

    const int stride = p.stride;

    for (int h = 0; h < n_head_kv; ++h) {
        // This scoring head reads the h-th query head and the h-th KV head. For GQA the
        // caller passes query heads already arranged so that scoring head h uses the first
        // query head of its group; an MSA-style broadcast handles the rest. Here (no GQA in
        // the reference harness) we index the h-th head of each buffer.
        std::vector<float> scores;
        antidiag_scores(Q + (size_t) h * (size_t) q_len * dim, q_len,
                        K + (size_t) h * (size_t) k_len * dim, k_len,
                        dim, stride, scores);

        std::vector<float> attn_sum;
        block_attn_sum(scores,
                       (q_len + stride-1)/stride, (k_len + stride-1)/stride,
                       q_len, k_len, p.block_size, stride, p.softmax,
                       n_q_blk, n_kv_blk, attn_sum);

        std::vector<uint8_t> mask;
        select_blocks(attn_sum, n_q_blk, n_kv_blk, p, mask);

        res.attn_sum.insert(res.attn_sum.end(), attn_sum.begin(), attn_sum.end());

        for (int qb = 0; qb < n_q_blk; ++qb) {
            std::vector<uint8_t> row(mask.begin() + (size_t) qb*n_kv_blk,
                                     mask.begin() + (size_t) qb*n_kv_blk + n_kv_blk);
            res.rows[(size_t) h*n_q_blk + qb] = std::move(row);
        }
    }
    return res;
}

} // namespace xattn
