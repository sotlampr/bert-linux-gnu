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

// FullTokenize, as in the original BERT implementation.
// Combines a BasicTokenizer and a FullTokenizer
class FullTokenizer {
  public:
    FullTokenizer(const std::string& vocabFname,
                  const std::string& lowercaseFname);

    // Detects if the model is using lowercase texts from the presence of a
    // placeholder file in the model directory
    bool getDoLowercase(const std::string& lowercaseFname) const;

    // Tokenize a sentence to icu::UnicodeString word pieces
    std::vector<icu::UnicodeString> tokenize(const std::string &s);

    // Tokenize a sentence and convert to ids using a vocabulary
    std::vector<long> tokenizeToIds (const std::string &s);

    // Get the id for a single wordpiece token
    long tokenToId(const icu::UnicodeString &s) const;

    ~FullTokenizer();
  private:
    Vocabulary readVocabulary(const std::string& vocabFname);
    std::map<UnicodeString, long> vocab;
    std::map<long, UnicodeString> invVocab;
    UErrorCode uErr = U_ZERO_ERROR;
    UnicodeConverter &unicoder;
    const BasicTokenizer &basicTokenizer;
    WordPieceTokenizer &wordPieceTokenizer;
};
#endif
