#ifndef TOKENIZE_H
#define TOKENIZE_H
#include <fstream>
#include <iostream>
#include <cassert>
#include <map>
#include <vector>
#include <iterator>

#include <unicode/ustream.h>
#include <unicode/schriter.h>
#include <unicode/brkiter.h>
#include <unicode/normalizer2.h>


// Convert to icu::UnicodeString and perform 'NFD' normalization
class UnicodeConverter {
  public:
    explicit UnicodeConverter(UErrorCode &errorCode);
    UnicodeString toUnicode(const std::string &s) const;
    UnicodeString process(const std::string &s, UErrorCode &errorCode) const;
  private:
    const Normalizer2 &nfd;
};

// Basic tokenizer: Splits punctuation, CJK chars and whitecpace.
class BasicTokenizer {
  public:
    explicit BasicTokenizer(bool doLowerCase);
    std::vector<icu::UnicodeString> tokenize(icu::UnicodeString &s) const;
  private:
    const bool doLowerCase;
    // Invalid character removal and whitespace cleanup
    icu::UnicodeString
      clean(const icu::UnicodeString &i) const;

    // Add whitespace between CJK characters
    icu::UnicodeString
      tokenizeCJKChars(const icu::UnicodeString &i) const;

    // Split by whitespace
    std::vector<icu::UnicodeString>
      whitespaceTokenize(const icu::UnicodeString &s) const;

    //  Strip 'Nm' category unicode accents
    icu::UnicodeString
      stripAccents(const icu::UnicodeString &i) const;

    // Split at punctuation 
    std::vector<icu::UnicodeString>
      splitPunctuation(const icu::UnicodeString s) const;
};

// WordPiece tokenizer: longest-match-first tokenization given a vocabulary
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

// Basic- and WordPiece tokenization (also converts to unicode and normalizes)
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
