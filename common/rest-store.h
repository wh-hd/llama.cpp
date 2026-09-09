#pragma once
//
// common/rest-store.h: REST (Retrieval-Based Speculative Decoding) datastore.
//
// ref: https://github.com/FasterDecoding/REST
//
// The datastore holds an OFFICELL token corpus (a flat file of token ids).
// At inference, the current context suffix is used as a retrieval key to find
// a document that ends with the same suffix, and its continuation is returned
// as draft tokens.
//
// M1: linear n-gram lookup over the loaded corpus (no suffix array yet).
// M2: replace with a suffix-array index + mmap for sub-second retrieval.
//

#include "llama.h"
#include "common.h"

#include <cstdint>
#include <string>
#include <vector>

// Offline token-corpus datastore used as the REST draft source.
struct common_rest_store {
    // flattened token corpus
    std::vector<llama_token> corpus;

    // M2: suffix array over `corpus` (int32 positions), built with libsais.
    // Sorted suffixes; each entry is the start index of a suffix in `corpus`.
    std::vector<int32_t> sa;

    // search params (copied from common_params_speculative_rest at load)
    uint16_t size_key   = 8;   // max retrieval key length (tried down to 2)
    uint16_t size_draft = 16;  // max draft tokens to return
    uint16_t topk       = 4;   // number of candidate continuations per key

    // Returns true if the datastore was loaded successfully.
    // The file format is a sequence of little-endian u32 token ids.
    // After loading, `build_sa()` runs to index the corpus.
    bool load(const std::string & path);

    // (Re)builds the suffix array over `corpus` using libsais. True on success.
    bool build_sa();

    // Retrieves up to `topk` continuations of the given key suffix.
    // `key` is the retrieval key (the last `size_key` tokens of the context).
    // Each returned entry is a continuation (draft) token sequence.
    // On success returns the top-k draft sequences; on no match returns empty.
    std::vector<std::vector<llama_token>> retrieve(const llama_tokens & key, int topk) const;

    // M3 (tree): retrieves ALL distinct continuations of `key` (not capped by
    // topk; each capped by size_draft length). Used to build the candidate
    // Trie. Returns empty on no match.
    std::vector<std::vector<llama_token>> retrieve_all(const llama_tokens & key) const;
};
