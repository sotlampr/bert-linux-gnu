#ifndef WORDPIECE_TOKENIZER_H
#define WORDPIECE_TOKENIZER_H
#include <map>
#include <string>
#include <vector>

#include <unicode/ustream.h>

// WordPieceTokenizer, as in the original BERT implementation.
// Expects icu::UnicodeString(s)
class WordPieceTokenizer {
  public:
    WordPieceTokenizer(const std::map<icu::UnicodeString,long> &vocab,
                       const icu::UnicodeString &unkToken,
                       int maxInputCharsPerWord);
    // Convert a sentence to tokens
    std::vector<icu::UnicodeString> tokenize(const icu::UnicodeString &s) const;

    // Convert tokens to ids using the vocabulary
    std::vector<long> tokensToIds(const std::vector<icu::UnicodeString> &s) const;

    // Get the id for a single wordpiece token
    long tokenToId(const icu::UnicodeString &s) const;
  private:
    std::map<UnicodeString, long> vocab;
    const icu::UnicodeString unkToken;
    const int maxInputCharsPerWord;
};
#endif
