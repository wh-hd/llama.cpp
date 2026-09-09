#include "rest-store.h"
#include "log.h"
#include "libsais.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <algorithm>

// Load a flat corpus file of little-endian u32 token ids.
bool common_rest_store::load(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        LOG_ERR("%s: failed to open datastore '%s'\n", __func__, path.c_str());
        return false;
    }

    corpus.clear();
    sa.clear();
    uint32_t id;
    while (f.read(reinterpret_cast<char *>(&id), sizeof(id))) {
        corpus.push_back((llama_token) id);
    }

    if (corpus.empty()) {
        LOG_ERR("%s: datastore '%s' is empty\n", __func__, path.c_str());
        return false;
    }

    if (!build_sa()) {
        LOG_ERR("%s: failed to build suffix array for '%s'\n", __func__, path.c_str());
        return false;
    }

    LOG_INF("%s: loaded datastore '%s' with %zu tokens (suffix array: %zu)\n",
            __func__, path.c_str(), corpus.size(), sa.size());
    return true;
}

// Build the suffix array over `corpus` using libsais (the same library the
// reference REST implementation uses). We suffix all positions [0, n), and
// index over the integer alphabet [0, k) where k = max token id + 1. Note that
// libsais requires a sentinel: the last element T[n-1] must be unique and less
// than every other symbol. To keep the suffix array well-formed over our token
// ids (which can be large), we build it over a transformed buffer where we
// append a distinct sentinel and rank keep values. The simplest correct route:
// pass the raw token buffer to libsais_int with k = n (positions) is NOT right.
// Instead we mirror what the reference does: libsais_int(buffer, SA, n, vocab_size, 0)
// -- the reference assumes the last token in each dumped chunk is a sentinel
// (they comment "// self.buffer.push(34999);" and use vocab_size). To be robust
// for arbitrary corpora we append our own sentinel and indicate it via `fs=0`
// and using a lexicographic transform. Pragmatically: for token ids < 2^31 we
// keep a copy, append a sentinel = max+1 that is larger than all (so it sorts
// last and never participates in a key), then build over [0, n) without the
// sentinel position being a valid match start (we trim it below).
bool common_rest_store::build_sa() {
    const size_t n = corpus.size();
    if (n < 1) {
        return false;
    }

    // copy corpus into int32 buffer (token ids are < 2^31)
    std::vector<int32_t> T(n + 1);
    int32_t max_token = 0;
    for (size_t i = 0; i < n; ++i) {
        T[i] = (int32_t) corpus[i];
        if (T[i] > max_token) {
            max_token = T[i];
        }
    }
    // sentinel value: strictly greater than every token so its suffix sorts
    // last and is never returned as a match (no key contains it).
    T[n] = max_token + 1;

    sa.assign(n + 1, 0);

    // libsais_int(T, SA, n, k, fs): builds SA for the length-(n+1) buffer.
    const int32_t len = (int32_t)(n + 1);
    const int32_t k   = max_token + 2; // alphabet size covers all tokens + sentinel
    const int32_t ret = libsais_int(T.data(), sa.data(), len, k, /*fs=*/0);
    if (ret != 0) {
        LOG_ERR("%s: libsais_int returned %d\n", __func__, ret);
        sa.clear();
        return false;
    }

    return true;
}

// Binary search over the suffix array for the lexicographic interval [lo, hi)
// of suffixes that start with `key`. Uses the classic two-bound approach.
static void sa_lower_bound(
        const std::vector<llama_token> & corpus,
        const std::vector<int32_t> & sa,
        const llama_tokens & key,
        size_t & lo, size_t & hi) {
    const size_t n = sa.size();
    const size_t kn = key.size();
    lo = 0;
    hi = n;

    auto cmp_suffix = [&](size_t pos, const llama_tokens & k) -> int {
        // compare corpus[sa[pos] ...] with k[0..]
        size_t p = (size_t) sa[pos];
        const size_t avail = corpus.size() - p;
        const size_t lim = std::min(avail, k.size());
        for (size_t i = 0; i < lim; ++i) {
            llama_token a = corpus[p + i];
            llama_token b = k[i];
            if (a != b) {
                return a < b ? -1 : 1;
            }
        }
        // shorter suffix (lim==avail<k.size) sorts before
        if (lim < k.size()) {
            return -1;
        }
        return 0;
    };

    // first suffix >= key
    size_t l = 0, r = n;
    while (l < r) {
        size_t m = l + (r - l) / 2;
        if (cmp_suffix(m, key) < 0) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    lo = l;

    // first suffix > key
    l = lo; r = n;
    while (l < r) {
        size_t m = l + (r - l) / 2;
        if (cmp_suffix(m, key) <= 0) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    hi = l;
}

// Retrieve draft continuations via the suffix array (M2).
// Finds every suffix starting with `key`, takes the `long` tokens after each
// match as a draft segment, dedupes, and returns up to `topk`.
// Shared retrieval core: returns ALL distinct continuations of `key`
// (each capped by size_draft), in suffix-array (lexicographic) order.
// Implemented in an anonymous namespace so both public methods share it.
namespace {

std::vector<std::vector<llama_token>> sa_retrieve_all(
        const std::vector<llama_token> & corpus,
        const std::vector<int32_t> & sa,
        const llama_tokens & key,
        uint16_t size_draft) {
    std::vector<std::vector<llama_token>> candidates;

    const size_t kn = key.size();
    if (kn == 0 || sa.empty() || corpus.size() <= kn) {
        return candidates;
    }

    const size_t max_draft = size_draft;

    size_t lo = 0, hi = 0;
    sa_lower_bound(corpus, sa, key, lo, hi);

    for (size_t idx = lo; idx < hi; ++idx) {
        const size_t p = (size_t) sa[idx];
        const size_t start = p + kn;
        if (start >= corpus.size()) {
            continue; // suffix is exactly the key with nothing after
        }
        const size_t avail = std::min(max_draft, corpus.size() - start);
        if (avail == 0) {
            continue;
        }
        std::vector<llama_token> cont(corpus.begin() + start, corpus.begin() + start + avail);

        bool dup = false;
        for (const auto & c : candidates) {
            if (c == cont) {
                dup = true;
                break;
            }
        }
        if (dup) {
            continue;
        }
        candidates.push_back(std::move(cont));
    }

    return candidates;
}

} // namespace

std::vector<std::vector<llama_token>> common_rest_store::retrieve(const llama_tokens & key, int topk) const {
    auto all = sa_retrieve_all(corpus, sa, key, size_draft);
    if (topk <= 0 || (int) all.size() <= topk) {
        return all;
    }
    all.resize((size_t) topk);
    return all;
}

std::vector<std::vector<llama_token>> common_rest_store::retrieve_all(const llama_tokens & key) const {
    return sa_retrieve_all(corpus, sa, key, size_draft);
}

