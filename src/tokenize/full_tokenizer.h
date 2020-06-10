#ifndef FULL_TOKENIZER_H
#define FULL_TOKENIZER_H
#include <map>
#include <string>
#include <vector>

#include <unicode/ustream.h>

#include "basic_tokenizer.h"
#include "unicode_converter.h"
#include "wordpiece_tokenizer.h"

using Vocabulary = std::pair<std::map<icu::UnicodeString,long>,std::map<long,icu::UnicodeString>>;

class FullTokenizer {
  public:
    explicit FullTokenizer(const std::string &modelDir);
    bool getDoLowercase(const std::string& modelDir) const;
    Vocabulary readVocabulary(const std::string &modelDir);
    std::vector<icu::UnicodeString> tokenize(const std::string &s);
    std::vector<long> tokenizeToIds (const std::string &s);
    long tokenToId(const icu::UnicodeString &s) const;
    ~FullTokenizer();
  private:
    std::map<UnicodeString, long> vocab;
    std::map<long, UnicodeString> invVocab;
    UErrorCode uErr = U_ZERO_ERROR;
    UnicodeConverter &unicoder;
    const BasicTokenizer &basicTokenizer;
    WordPieceTokenizer &wordPieceTokenizer;
};
#endif
