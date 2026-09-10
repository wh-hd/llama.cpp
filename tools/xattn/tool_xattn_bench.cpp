// tool_xattn_bench.cpp - Pure-attention benchmark: dense FA vs XAttention compact FA.
//
// Measures ONLY the attention computation (QK^T/FA), NOT the weight matmuls. This mirrors how
// XAttention's 13.5x is reported (attention-compute speedup), decoupled from the 9B weight ops.
//
// Build & run (single stream, CPU for determinism):
//   g++ -std=c++17 -O2 -I ggml/include -I src -I . tools/xattn/tool_xattn_bench.cpp \
//       -o /tmp/xattn/xbench -L build-master/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,build-master/bin
//   /tmp/xattn/xbench <n_kv> <n_tok> <blk> <K_eff> <D> <HKV> <reps>
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

static ggml_tensor * tl(ggml_context * ctx, ggml_type type, int n0, int n1) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, n0, n1);
    ggml_set_name(t, "tmp");
    return t;
}

int main(int argc, char ** argv) {
    const int64_t n_kv   = argc > 1 ? atoll(argv[1]) : 2048;
    const int64_t n_tok  = argc > 2 ? atoll(argv[2]) : 512;
    const int64_t blk    = argc > 3 ? atoll(argv[3]) : 16;
    const int64_t K_eff  = argc > 4 ? atoll(argv[4]) : 16;   // selected blocks
    const int64_t D      = argc > 5 ? atoll(argv[5]) : 128;
    const int64_t HKV    = argc > 6 ? atoll(argv[6]) : 4;
    const int      reps  = argc > 7 ? atoi(argv[7]) : 20;
    const int64_t n_kv_blk = (n_kv + blk - 1) / blk;
    const float scale = 1.0f/std::sqrt((float)D);

    ggml_init_params ipar = { 1u<<28, nullptr, false };
    ggml_context * ctx = ggml_init(ipar);
    if (!ctx) { fprintf(stderr, "no ctx\n"); return 2; }

    // ---- inputs: q [D, n_tok*HQ?] use HQ=HKV for simplicity; k [D, n_kv] ----
    const int64_t HQ = HKV;
    ggml_tensor * q = tl(ctx, GGML_TYPE_F16, D, n_tok);
    ggml_tensor * k = tl(ctx, GGML_TYPE_F16, D, n_kv);
    ggml_tensor * v = tl(ctx, GGML_TYPE_F16, D, n_kv);

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) { fprintf(stderr, "no cpu backend\n"); return 2; }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu);
    ggml_gallocr_t gallo = ggml_gallocr_new(buft);

    // ---- graph 1: DENSE attention [n_tok x n_kv] ----
    ggml_context * cd = ggml_init(ipar);
    ggml_tensor * qd = tl(cd, GGML_TYPE_F16, D, n_tok);
    ggml_tensor * kd = tl(cd, GGML_TYPE_F16, D, n_kv);
    ggml_tensor * vd = tl(cd, GGML_TYPE_F16, D, n_kv);
    // FA expects 4D: q [D,n_tok,1,1], k/v [D,n_kv,1,1]
    ggml_tensor * qd4 = ggml_view_4d(cd, qd, D, n_tok, 1, 1, D*2, D*n_tok*2, D*n_tok*2, 0);
    ggml_tensor * kd4 = ggml_view_4d(cd, kd, D, n_kv,  1, 1, D*2, D*n_kv*2,  D*n_kv*2,  0);
    ggml_tensor * vd4 = ggml_view_4d(cd, vd, D, n_kv,  1, 1, D*2, D*n_kv*2,  D*n_kv*2,  0);
    ggml_tensor * od  = ggml_flash_attn_ext(cd, qd4, kd4, vd4, nullptr, scale, 0, 0);
    ggml_cgraph * gd = ggml_new_graph(cd);
    ggml_build_forward_expand(gd, od);
    ggml_gallocr_alloc_graph(gallo, gd);
    // NOTE: inputs left zero-filled; timing is input-independent.

    // ---- graph 2: XATTN compact [n_tok x blk*K_eff], selecting recent K_eff blocks ----
    ggml_context * cx = ggml_init(ipar);
    ggml_tensor * qx = tl(cx, GGML_TYPE_F16, D, n_tok);
    ggml_tensor * kx = tl(cx, GGML_TYPE_F16, D, n_kv);
    ggml_tensor * vx = tl(cx, GGML_TYPE_F16, D, n_kv);
    const int64_t n_sel = blk*K_eff;
    // index of selected recent tokens: tokens [n_kv-n_sel, n_kv)
    std::vector<int32_t> sel; sel.reserve(n_sel);
    for (int64_t i = n_kv-n_sel; i < n_kv; ++i) sel.push_back((int32_t)i);
    ggml_tensor * kidx = ggml_new_tensor_1d(cx, GGML_TYPE_I32, n_sel);
    ggml_tensor * kc = ggml_get_rows(cx, kx, kidx);          // [D, n_sel]
    ggml_tensor * vc = ggml_get_rows(cx, vx, kidx);          // [D, n_sel]
    ggml_tensor * qx4 = ggml_view_4d(cx, qx, D, n_tok, 1, 1, D*2, D*n_tok*2, D*n_tok*2, 0);
    ggml_tensor * kc4 = ggml_view_4d(cx, kc, D, n_sel, 1, 1, D*2, D*n_sel*2, D*n_sel*2, 0);
    ggml_tensor * vc4 = ggml_view_4d(cx, vc, D, n_sel, 1, 1, D*2, D*n_sel*2, D*n_sel*2, 0);
    ggml_tensor * ox  = ggml_flash_attn_ext(cx, qx4, kc4, vc4, nullptr, scale, 0, 0);
    ggml_cgraph * gx = ggml_new_graph(cx);
    ggml_build_forward_expand(gx, ox);
    ggml_gallocr_alloc_graph(gallo, gx);
    // fill the gather index directly (CPU backend exposes writable host data after alloc)
    memcpy(kidx->data, sel.data(), (size_t)n_sel*4);
    // inputs left zero-filled (timing input-independent)
    ggml_backend_graph_compute(cpu, gx);

    // ---- time both ----
    auto t0 = ggml_time_us();
    for (int r = 0; r < reps; ++r) ggml_backend_graph_compute(cpu, gd);
    double td = (ggml_time_us()-t0)/1e6/reps;
    t0 = ggml_time_us();
    for (int r = 0; r < reps; ++r) ggml_backend_graph_compute(cpu, gx);
    double tx = (ggml_time_us()-t0)/1e6/reps;

    const double dense_flops = 2.0*n_tok*n_kv*D;                 // QK^T (approx, excludes AV)
    const double comp_flops  = 2.0*n_tok*(blk*K_eff)*D;
    printf("n_kv=%lld n_tok=%lld blk=%lld K=%lld -> compact=%lld (%.2fx fewer KV)\n",
           n_kv, n_tok, blk, K_eff, n_sel, (double)n_kv/n_sel);
    printf("dense : %8.4f ms  (%.4g GFLOP)\n", td*1e3, dense_flops/1e9);
    printf("xattn : %8.4f ms  (%.4g GFLOP)\n", tx*1e3, comp_flops/1e9);
    printf("dense/xattn time ratio = %.3fx\n", td/tx);

    ggml_free(cx); ggml_free(cd); ggml_free(ctx);
    return 0;
}
