#ifndef BASIC_TOKENIZER_H
#define BASIC_TOKENIZER_H
#include <string>
#include <vector>

#include <unicode/ustream.h>

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
#endif
