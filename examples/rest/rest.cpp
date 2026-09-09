//
// examples/rest/rest.cpp
//
// REST (Retrieval-Based Speculative Decoding) - tree verification demo.
//
// Single target model + offline token-corpus datastore. The datastore is
// queried by the current context suffix; the top candidate continuations form
// a TRIE of draft branches. Each root->leaf trie path becomes its own seq in
// the target batch (sharing the confirmed KV prefix with seq 0), so ONE
// forward verifies ALL draft tokens. We greedily accept the LONGEST consistent
// path. seq0's KV monotonically grows as the accepted chain (satisfies M-RoPE).
//
// Mirrors the KV-management of examples/speculative/speculative.cpp.
//

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "rest-store.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct seq_draft {
    bool active = false;

    std::vector<int>         i_batch_tgt; // batch slot of each drafted token
    std::vector<llama_token> tokens;      // drafted token sequence (the path)
};

// Build a draft tree from the datastore continuation set.
// tree_nodes[i] = { token id, parent node }, node 0 = root placeholder.
static void build_tree(const common_rest_store & store, const llama_tokens & key,
                       int n_draft,
                       std::vector<std::pair<llama_token, int32_t>> & tree_nodes) {
    tree_nodes.clear();
    auto conts = store.retrieve_all(key);
    if (conts.empty()) {
        return;
    }
    tree_nodes.push_back({ (llama_token) -1, -1 }); // root

    for (const auto & cont : conts) {
        int32_t parent = 0;
        for (llama_token id : cont) {
            if ((int32_t) tree_nodes.size() >= n_draft + 1) break;
            int32_t child = -1;
            for (size_t i = 1; i < tree_nodes.size(); ++i) {
                if (tree_nodes[i].second == parent && tree_nodes[i].first == id) {
                    child = (int32_t) i;
                    break;
                }
            }
            if (child == -1) {
                child = (int32_t) tree_nodes.size();
                tree_nodes.push_back({ id, parent });
            }
            parent = child;
        }
    }
    if (tree_nodes.size() == 1) {
        tree_nodes.clear();
    }
}

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.speculative.rest.datastore_path.empty()) {
        LOG_ERR("%s: --spec-rest-datastore is required\n", __func__);
        return 1;
    }

    const int n_seq_dft = std::max(2, params.n_parallel);  // tree branches
    const int n_draft   = params.speculative.rest.size_draft;

    const auto output_limits = common_speculative_get_output_limits(
            params.n_batch, n_seq_dft, n_draft);
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    common_rest_store store;
    store.size_key   = params.speculative.rest.size_key;
    store.size_draft = params.speculative.rest.size_draft;
    store.topk       = params.speculative.rest.topk;
    if (!store.load(params.speculative.rest.datastore_path)) {
        LOG_ERR("%s: failed to load datastore\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init_tgt = common_init_from_params(params);

    llama_model * model_tgt = llama_init_tgt->model();
    if (model_tgt == NULL) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
        return 1;
    }
    llama_context * ctx_tgt = llama_init_tgt->context();
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);

    auto * mem_tgt = llama_get_memory(ctx_tgt);

    std::string prompt_text = params.prompt;
    const bool is_interactive = params.interactive && !params.prompt.empty();
    if (prompt_text.empty()) {
        LOG_ERR("%s: --prompt is required\n", __func__);
        return 1;
    }

    std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompt_text, true, true);

    const int n_input = (int) inp.size();
    if (n_input == 0) {
        LOG_ERR("%s: empty prompt\n", __func__);
        return 1;
    }

    LOG("\n\n");
    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    // decode all but the last prompt token; keep the last token separate
    if (llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), n_input - 1))) {
        LOG_ERR("%s: failed to eval prompt\n", __func__);
        return 1;
    }

    // confirmed KV seq0 content = inp[0..n_input-2]; id_last = inp.back()
    // Invariant: cur_tokens == seq0 KV content (confirmed chain, EXCLUDING the
    // current id_last endpoint). n_past_tgt == cur_tokens.size().
    std::vector<llama_token> cur_tokens(inp.begin(), inp.end() - 1);
    llama_token id_last = inp.back();
    int n_past_tgt = (int) cur_tokens.size();

    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);

    const auto t_dec_start = ggml_time_us();

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;
    bool has_eos  = false;

    while (true) {
        // ---- build the draft tree from the datastore (conditioned on context) ----
        // context for retrieval = confirmed (cur_tokens) + id_last
        std::vector<llama_token> ctx_tokens = cur_tokens;
        ctx_tokens.push_back(id_last);

        llama_tokens key;
        for (int32_t span = store.size_key; span >= 2; --span) {
            if ((int) ctx_tokens.size() < span) continue;
            key.clear();
            for (int i = (int) ctx_tokens.size() - span; i < (int) ctx_tokens.size(); ++i) {
                key.push_back(ctx_tokens[i]);
            }
            if (!store.retrieve_all(key).empty()) break;
        }

        std::vector<std::pair<llama_token, int32_t>> tree_nodes;
        build_tree(store, key, n_draft, tree_nodes);

        // ---- branch paths: one seq per root->node trie path ----
        std::vector<std::vector<llama_token>> paths;
        for (size_t i = 1; i < tree_nodes.size(); ++i) {
            std::vector<llama_token> p;
            int32_t cur = (int32_t) i;
            while (cur > 0) { p.push_back(tree_nodes[cur].first); cur = tree_nodes[cur].second; }
            std::reverse(p.begin(), p.end());
            paths.push_back(p);
        }
        std::sort(paths.begin(), paths.end());
        paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
        std::sort(paths.begin(), paths.end(),
                  [](const auto & a, const auto & b) { return a.size() > b.size(); });
        if ((int) paths.size() > n_seq_dft - 1) paths.resize(n_seq_dft - 1);

        std::vector<seq_draft> drafts(n_seq_dft);
        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active = false;
            drafts[s].tokens.clear();
            drafts[s].i_batch_tgt.clear();
        }

        // ---- build the tree batch ----
        // seq0: id_last at position n_past_tgt (monotonic, > KV seq0 max)
        common_batch_clear(batch_tgt);
        drafts[0].active = true;
        drafts[0].i_batch_tgt.push_back(0);
        common_batch_add(batch_tgt, id_last, n_past_tgt, { 0 }, true);

        int n_seq_cur = 1;
        for (size_t ip = 0; ip < paths.size() && n_seq_cur < n_seq_dft; ++ip) {
            const int s = n_seq_cur++;
            drafts[s].active = true;
            drafts[s].tokens = paths[ip];
            for (size_t d = 0; d < paths[ip].size(); ++d) {
                drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);
                // each branch seq's KV = seq0 copy + its own path
                common_batch_add(batch_tgt, paths[ip][d], n_past_tgt + 1 + d, { s }, true);
                ++n_drafted;
            }
        }

        // ---- share KV prefix (seq0 := confirmed chain) with all branch seqs ----
        for (int s = 1; s < n_seq_cur; ++s) {
            llama_memory_seq_cp(mem_tgt, 0, s, -1, -1);
        }
        if (llama_decode(ctx_tgt, batch_tgt)) {
            LOG_ERR("%s: failed to eval tree batch\n", __func__);
            return 1;
        }
        ++n_past_tgt;

        // ---- greedy acceptance (deterministic argmax, matches REST evaluate_posterior) ----
        auto argmax_at = [&](int slot) -> llama_token {
            const float * logits = llama_get_logits_ith(ctx_tgt, slot);
            GGML_ASSERT(logits != nullptr && "logits slot invalid");
            const int n_vocab = llama_vocab_n_tokens(vocab_tgt);
            int best = 0;
            float best_v = logits[0];
            for (int v = 1; v < n_vocab; ++v) {
                if (logits[v] > best_v) { best_v = logits[v]; best = v; }
            }
            return (llama_token) best;
        };

        // slot0 (id_last) predicts the anchor token
        llama_token anchor = argmax_at(0);

        // For each path, accept tokens matching argmax of the parent's slot.
        int best_accept = 0;
        std::vector<llama_token> best_branch;
        for (int s = 1; s < n_seq_cur; ++s) {
            std::vector<llama_token> chain;
            int prev_slot = 0;
            for (size_t q = 0; q < drafts[s].tokens.size(); ++q) {
                llama_token pred = argmax_at(prev_slot);
                if (pred != drafts[s].tokens[q]) break;
                chain.push_back(drafts[s].tokens[q]);
                prev_slot = drafts[s].i_batch_tgt[q];
            }
            if ((int) chain.size() > best_accept) {
                best_accept = (int) chain.size();
                best_branch = chain;
            }
        }

        // find the winning seq for KV collapse
        int best_s = -1;
        for (int s = 1; s < n_seq_cur; ++s) {
            if (best_accept > 0 &&
                (int) drafts[s].tokens.size() >= best_accept &&
                std::equal(drafts[s].tokens.begin(),
                           drafts[s].tokens.begin() + best_accept,
                           best_branch.begin())) {
                best_s = s;
                break;
            }
        }

        // ---- commit: anchor + accepted branch become the new confirmed chain ----
        LOG("%s", common_token_to_piece(ctx_tgt, anchor).c_str());
        ++n_predict;
        if (llama_vocab_is_eog(llama_model_get_vocab(model_tgt), anchor)) has_eos = true;

        std::vector<llama_token> accepted;
        if (best_accept > 0) {
            for (llama_token t : best_branch) {
                LOG("%s", common_token_to_piece(ctx_tgt, t).c_str());
                accepted.push_back(t);
                if (llama_vocab_is_eog(llama_model_get_vocab(model_tgt), t)) has_eos = true;
                ++n_accept;
                ++n_predict;
            }
        }

        // ---- collapse KV: seq0 = confirmed chain, discard branch seqs ----
        // The confirmed chain grows by [id_last, anchor, ...accepted]. We need
        // seq0 KV to hold exactly that. Two cases:
        //  - accepted: the winning branch seq holds id_last+anchor+accepted KV;
        //    copy it into seq0.
        //  - not accepted: only [id_last, anchor] are new; anchor was argmax of
        //    id_last (not decoded into a branch). Decode anchor into seq0.
        if (best_accept > 0) {
            llama_memory_seq_cp(mem_tgt, best_s, 0, -1, -1);
            int new_n = n_past_tgt + 1 + best_accept; // + id_last + accepted
            llama_memory_seq_rm(mem_tgt, 0, new_n, -1);
        } else {
            // decode anchor into seq0 (position n_past_tgt+1)
            llama_memory_seq_keep(mem_tgt, 0);
            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, anchor, n_past_tgt + 1, { 0 }, true);
            if (llama_decode(ctx_tgt, batch_tgt)) {
                LOG_ERR("%s: failed to eval anchor\n", __func__);
                return 1;
            }
        }
        llama_memory_seq_keep(mem_tgt, 0);
        for (int s = 1; s < n_seq_cur; ++s) {
            llama_memory_seq_rm(mem_tgt, s, -1, -1);
        }

        // ---- update state ----
        cur_tokens.push_back(id_last);
        cur_tokens.push_back(anchor);
        for (llama_token t : accepted) {
            cur_tokens.push_back(t);
        }
        id_last = cur_tokens.back();
        cur_tokens.pop_back(); // exclude newest endpoint from cur_tokens
        // n_past_tgt = seq0 KV max position + 1 (robust to KV trimming/copy)
        n_past_tgt = (int) llama_memory_seq_pos_max(mem_tgt, 0) + 1;
        if (n_past_tgt < (int) cur_tokens.size()) {
            n_past_tgt = (int) cur_tokens.size();
        }

        if (is_interactive) {
            LOG("\n");
        }

        if ((params.n_predict >= 0 && n_predict >= params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n",
            n_predict, (t_dec_end - t_dec_start) / 1e6f,
            n_predict / ((t_dec_end - t_dec_start) / 1e6f));
    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / (n_drafted > 0 ? n_drafted : 1));

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    llama_batch_free(batch_tgt);
    llama_free(ctx_tgt);
    llama_model_free(model_tgt);
    llama_backend_free();

    LOG("\n\n");
    return 0;
}
