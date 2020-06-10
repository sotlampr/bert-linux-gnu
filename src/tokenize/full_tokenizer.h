#ifndef FULL_TOKENIZER_H
#define FULL_TOKENIZER_H
#include <map>
#include <string>
#include <vector>

#include <unicode/ustream.h>

#include "basic_tokenizer.h"
#include "unicode_converter.h"
#include "wordpiece_tokenizer.h"

class FullTokenizer {
  public:
    FullTokenizer(const std::string &vocabFile, bool doLowerCase);
    std::pair<std::map<icu::UnicodeString,long>,std::map<long,icu::UnicodeString>>
      readVocabulary(const std::string &vocabFile);
    std::vector<icu::UnicodeString> tokenize(const std::string &s);
    std::vector<long> tokenizeToIds (const std::string &s);
    long tokenToId(const icu::UnicodeString &s) const;
    ~FullTokenizer();
  private:
    std::map<UnicodeString, long> vocab;
    std::map<long, UnicodeString> invVocab;
    UErrorCode uErr = U_ZERO_ERROR;
    const bool doLowerCase;
    UnicodeConverter &unicoder;
    const BasicTokenizer &basicTokenizer;
    WordPieceTokenizer &wordPieceTokenizer;
};
#endif
