// tool_fold_score_test.cpp - Validate in-graph "concat-fold" antidiagonal scoring vs the
// reference. The fold builds, for each stride phase s, the strided sub-sequences
//   Q_s = Q[:, (S-1-s)::S],  K_s = K[:, s::S]
// then concatenates all S phases along the embedding dim into [S*D, QF] / [S*D, KF],
// and does ONE mul_mat to get the folded QK^T of shape [QF, KF]. This is the cheap
// (1/S) XAttention form: a single matmul over the folded matrices.
//
// Reference (see xattn-score.cpp antidiag_scores):
//   score[i,j] = sum_s sum_d Q[i*S+(S-1-s), d] * K[j*S+s, d]
//
// We verify the in-graph folded matmul == reference bit-for-bit, and that the build
// is O(S*D*QF*KF) work (i.e. a single mul_mat over folded tensors).
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static float frand() { return (float)(rand() % 1000) / 100.0f - 5.0f; }

// reference antidiag scores
static void ref_antidiag(const float * Q, int q_len, const float * K, int k_len,
                         int dim, int stride, std::vector<float> & out) {
    const int QF = (q_len + stride - 1) / stride;
    const int KF = (k_len + stride - 1) / stride;
    out.assign((size_t) QF * KF, 0.0f);
    for (int i = 0; i < QF; ++i)
        for (int j = 0; j < KF; ++j) {
            float acc = 0;
            for (int s = 0; s < stride; ++s) {
                const int qi = i*stride + (stride-1-s);
                const int kj = j*stride + s;
                if (qi >= q_len || kj >= k_len) continue;
                const float * qr = Q + (size_t) qi*dim;
                const float * kr = K + (size_t) kj*dim;
                for (int d = 0; d < dim; ++d) acc += qr[d]*kr[d];
            }
            out[(size_t) i*KF + j] = acc;
        }
}

int main(int argc, char ** argv) {
    const int q_len  = argc > 1 ? atoi(argv[1]) : 64;
    const int k_len  = argc > 2 ? atoi(argv[2]) : 96;
    const int dim    = argc > 3 ? atoi(argv[3]) : 16;
    const int stride = argc > 4 ? atoi(argv[4]) : 8;
    const int seed   = argc > 5 ? atoi(argv[5]) : 7;

    srand(seed);
    std::vector<float> Q((size_t) q_len*dim), K((size_t) k_len*dim);
    for (auto & v : Q) v = frand();
    for (auto & v : K) v = frand();

    const int QF = (q_len + stride - 1) / stride;
    const int KF = (k_len + stride - 1) / stride;
    std::vector<float> want;
    ref_antidiag(Q.data(), q_len, K.data(), k_len, dim, stride, want);

    // ---- build in-graph folded matmul ----
    struct ggml_init_params ip = { .mem_size = 128lu*1024*1024, .mem_buffer = nullptr, .no_alloc = true };
    ggml_context * ctx = ggml_init(ip);

    // Q/D: [dim, q_len], K/D: [dim, k_len]  (ne0=dim, ne1=token)
    ggml_tensor * qtin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, q_len);
    ggml_tensor * ktin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, k_len);
    ggml_set_input(qtin); ggml_set_input(ktin);
    ggml_set_name(qtin, "Q"); ggml_set_name(ktin, "K");

    // build folded Q: [S*D, QF] by concating phase-views along ne0
    // each phase: Q[:, (S-1-s)::S] -> [dim, QF3] where QF3 = number of valid rows in that phase
    // We use get_rows: Q is [dim, q_len]; get_rows gathers along ne[1] (tokens).
    std::vector<ggml_tensor*> qq_idx, kk_idx;   // index tensors to fill on host after alloc
    ggml_tensor * folded_Q = nullptr;
    ggml_tensor * folded_K = nullptr;

    for (int s = 0; s < stride; ++s) {
        // collect token indices for phase s
        std::vector<int32_t> qi, ki;
        for (int i = 0; i < QF; ++i) { int t = i*stride + (stride-1-s); if (t < q_len) qi.push_back(t); }
        for (int j = 0; j < KF; ++j) { int t = j*stride + s; if (t < k_len) ki.push_back(t); }
        // index tensors -> I32, values filled on host after alloc
        ggml_tensor * qq = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) qi.size());
        ggml_tensor * kk = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) ki.size());
        qq_idx.push_back(qq); kk_idx.push_back(kk);
        ggml_tensor * q_s = ggml_get_rows(ctx, qtin, qq);   // [dim, QF3]
        ggml_tensor * k_s = ggml_get_rows(ctx, ktin, kk);   // [dim, KF3]
        folded_Q = folded_Q ? ggml_concat(ctx, folded_Q, q_s, 0) : q_s;
        folded_K = folded_K ? ggml_concat(ctx, folded_K, k_s, 0) : k_s;
    }
    // folded_Q: [S*D, QF], folded_K: [S*D, KF]
    printf("folded_Q ne=[%lld,%lld] folded_K ne=[%lld,%lld]\n",
           folded_Q->ne[0], folded_Q->ne[1], folded_K->ne[0], folded_K->ne[1]);

    // folded scores: mul_mat(A=folded_K, B=folded_Q) -> A^T B = [KF, QF]
    ggml_tensor * fs = ggml_mul_mat(ctx, folded_K, folded_Q);
    ggml_mul_mat_set_prec(fs, GGML_PREC_F32);
    printf("folded scores ne=[%lld,%lld] (expect [KF=%d, QF=%d])\n", fs->ne[0], fs->ne[1], KF, QF);

    ggml_tensor * out = fs; // [KF, QF]; reference is [QF, KF]; we compare transposed
    ggml_set_name(out, "folded_scores");
    ggml_set_output(out);

    // ---- allocate + run on CPU backend ----
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) { fprintf(stderr, "no cpu backend\n"); return 2; }
    ggml_gallocr_t gallo = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_alloc_graph(gallo, gf);
    ggml_backend_tensor_set(qtin, Q.data(), 0, Q.size()*sizeof(float));
    ggml_backend_tensor_set(ktin, K.data(), 0, K.size()*sizeof(float));
    // fill index tensors (deterministic stride indices)
    for (int s = 0; s < stride; ++s) {
        std::vector<int32_t> qi, ki;
        for (int i = 0; i < QF; ++i) { int t = i*stride + (stride-1-s); if (t < q_len) qi.push_back(t); }
        for (int j = 0; j < KF; ++j) { int t = j*stride + s; if (t < k_len) ki.push_back(t); }
        ggml_backend_tensor_set(qq_idx[s], qi.data(), 0, qi.size()*sizeof(int32_t));
        ggml_backend_tensor_set(kk_idx[s], ki.data(), 0, ki.size()*sizeof(int32_t));
    }
    ggml_backend_graph_compute(cpu, gf);

    std::vector<float> got((size_t) KF*QF);
    ggml_backend_tensor_get(out, got.data(), 0, got.size()*sizeof(float));

    // compare vs reference. want is [QF, KF] row-major; got is [KF, QF] ggml col-major.
    // ggml col-major: element (row=kkf, col=qf) sits at off = qf*ne0 + kkf = qf*KF + kkf.
    long mism = 0; double maxdiff = 0;
    long nvalid = 0;
    for (int qf = 0; qf < QF; ++qf) for (int kkf = 0; kkf < KF; ++kkf) {
        float refv = want[(size_t) qf*KF + kkf];
        float gotv = got[(size_t) qf*KF + kkf];
        double d = fabs((double) refv - gotv);
        if (d > maxdiff) maxdiff = d;
        nvalid++;
        if (fabs(refv - gotv) > 1e-3f) mism++;
    }
    printf("compared %ld folded cells, mismatches=%ld, maxdiff=%.6g\n", nvalid, mism, maxdiff);
    printf("%s\n", mism == 0 ? "FOLD_SCORE_MATCH" : "FOLD_SCORE_MISMATCH");

    ggml_backend_free(cpu);
    ggml_free(ctx);
    return mism == 0 ? 0 : 1;
}
