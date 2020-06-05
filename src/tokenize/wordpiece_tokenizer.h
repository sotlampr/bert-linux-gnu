#ifndef WORDPIECE_TOKENIZER_H
#define WORDPIECE_TOKENIZER_H
#include <map>
#include <string>
#include <vector>
#include <unicode/ustream.h>

class WordPieceTokenizer {
  public:
    WordPieceTokenizer(const std::map<icu::UnicodeString,long> &vocab,
                       const icu::UnicodeString &unkToken,
                       int maxInputCharsPerWord);
    std::vector<icu::UnicodeString> tokenize(const icu::UnicodeString &s) const;
    std::vector<long> tokensToIds(const std::vector<icu::UnicodeString> &s) const;
    long tokenToId(const icu::UnicodeString &s) const;
  private:
    std::map<UnicodeString, long> vocab;
    const icu::UnicodeString unkToken;
    const int maxInputCharsPerWord;
};
#endif
