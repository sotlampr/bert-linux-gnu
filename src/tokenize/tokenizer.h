#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <string>
#include <vector>

class Tokenizer {
  public:
    virtual std::vector<std::string> tokenize(const std::string &s) {};
    virtual std::vector<long> tokenizeToIds (const std::string &s) {};
    virtual long tokenToId(const std::string &s) const {};
};
#endif
