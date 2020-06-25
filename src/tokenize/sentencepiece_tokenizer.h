#ifndef SENTENCEPIECE_TOKENIZER_H
#define SENTENCEPIECE_TOKENIZER_H
#include <fstream>
#include <map>
#include <vector>

#include <sentencepiece_processor.h>

#include "tokenizer.h"

class SentencepieceTokenizer : public virtual Tokenizer {
  public:
    SentencepieceTokenizer(const std::string& modelFname,
                           const std::string& lowercaseFname);

    std::vector<std::string> tokenize(const std::string &s);
    std::vector<long> tokenizeToIds (const std::string &s);
    long tokenToId(const std::string &s) const;

    ~SentencepieceTokenizer();
  private:
    std::string handleCase(std::string s) const;
    static bool getDoLowercase(const std::string& lowercaseFname);
    sentencepiece::SentencePieceProcessor &sentencepieceProcessor;
    const bool doLowerCase;
};
#endif
