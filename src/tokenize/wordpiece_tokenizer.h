#ifndef WORDPIECE_TOKENIZER_H
#define WORDPIECE_TOKENIZER_H
#include <map>
#include <string>
#include <vector>

// WordPieceTokenizer, as in the original BERT implementation.
// Expects icu::UnicodeString(s)
class WordPieceTokenizer {
  public:
    WordPieceTokenizer(const std::map<std::string, long> &vocab,
                       const std::string &unkToken,
                       size_t maxInputCharsPerWord);
    // Convert a sentence to tokens
    std::vector<std::string> tokenize(const std::string &s) const;

    // Convert tokens to ids using the vocabulary
    std::vector<long> tokensToIds(const std::vector<std::string> &s) const;

    // Get the id for a single wordpiece token
    long tokenToId(const std::string &s) const;
  private:
    std::map<std::string, long> vocab;
    const std::string unkToken;
    const size_t maxInputCharsPerWord;
};
#endif
