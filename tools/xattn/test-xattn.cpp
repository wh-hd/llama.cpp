// test-xattn.cpp - standalone correctness driver for the XAttention block scorer.
// Generates deterministic Q/K data, runs score_prefill, and dumps the block mask
// to stdout + a .npy-friendly text file so it can be compared against the official
// numpy/python reference (reference_xattn.py).
//
// This is a developer verification tool, not part of the formal test suite.

#include "xattn-score.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

// deterministic xorshift RNG so C++ and numpy can reproduce identical data
static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static uint64_t rng_next() {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}
static float frand(float lo, float hi) {
    const uint64_t v = rng_next();
    const float u = (float)(v >> 40) / (float)(1u << 24); // [0,1)
    return lo + (hi - lo) * u;
}

int main(int argc, char ** argv) {
    // args: q_len k_len dim n_head_kv stride block_size threshold outfile
    const int q_len   = argc > 1 ? std::atoi(argv[1]) : 512;
    const int k_len   = argc > 2 ? std::atoi(argv[2]) : 512;
    const int dim     = argc > 3 ? std::atoi(argv[3]) : 64;
    const int n_head  = argc > 4 ? std::atoi(argv[4]) : 4;    // kv heads (scoring)
    const int stride  = argc > 5 ? std::atoi(argv[5]) : 8;
    const int blk     = argc > 6 ? std::atoi(argv[6]) : 128;
    const float thr   = argc > 7 ? (float)std::atof(argv[7]) : 0.9f;
    const std::string out = argc > 8 ? argv[8] : "/tmp/xattn_out.txt";
    const uint64_t seed = argc > 9 ? std::strtoull(argv[9], nullptr, 10) : 12345;

    rng_state = seed;

    // Q: [n_head*q_len, dim], K: [n_head*k_len, dim]
    std::vector<float> Q((size_t) n_head*q_len*dim);
    std::vector<float> K((size_t) n_head*k_len*dim);
    for (size_t i = 0; i < Q.size(); ++i) Q[i] = frand(-1.0f, 1.0f);
    for (size_t i = 0; i < K.size(); ++i) K[i] = frand(-1.0f, 1.0f);

    xattn::score_params p;
    p.stride = stride;
    p.block_size = blk;
    p.threshold = thr;
    p.causal = true;
    p.keep_sink = true;
    p.keep_recent = true;
    p.softmax = true;

    xattn::block_mask m = xattn::score_prefill(Q.data(), n_head, q_len, dim,
                                               K.data(), n_head, k_len, p);

    // dump: n_head, n_q_block, n_kv_block then per (head, q_block) row of 0/1
    std::vector<uint32_t> outm;
    for (size_t r = 0; r < m.rows.size(); ++r) {
        const std::vector<uint8_t> & row = m.rows[r];
        for (uint8_t v : row) outm.push_back(v);
    }

    FILE * f = std::fopen(out.c_str(), "wb");
    if (!f) { perror("open"); return 1; }
    std::fprintf(f, "%d %d %d %d %d\n", n_head, m.n_q_block, m.n_kv_block, (int) outm.size(), stride);
    for (uint32_t v : outm) std::fprintf(f, "%d ", (int) v);
    std::fprintf(f, "\n");
    std::fclose(f);

    // also dump raw attn_sum (block attention mass) per head for cross-check
    FILE * g = std::fopen((out + ".attn").c_str(), "wb");
    if (g) {
        std::fprintf(g, "%d %d %d\n", n_head, m.n_q_block, m.n_kv_block);
        for (float v : m.attn_sum) std::fprintf(g, "%.9g ", (double) v);
        std::fprintf(g, "\n");
        std::fclose(g);
    }

    // also echo head count kept per q_block (readable)
    std::printf("n_head=%d q_len=%d k_len=%d dim=%d stride=%d block=%d thr=%.2f\n",
                n_head, q_len, k_len, dim, stride, blk, thr);
    std::printf("n_q_block=%d n_kv_block=%d total_keep=", m.n_q_block, m.n_kv_block);
    long kept = 0;
    for (size_t r = 0; r < m.rows.size(); ++r)
        for (uint8_t v : m.rows[r]) kept += v;
    std::printf("%ld\n", kept);
    std::printf("wrote %s\n", out.c_str());
    return 0;
}
