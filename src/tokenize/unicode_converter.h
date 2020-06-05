#ifndef UNICODE_CONVERTER_H
#define UNICODE_CONVERTER_H
#include <string>
#include <unicode/ustream.h>
#include <unicode/normalizer2.h>

class UnicodeConverter {
  public:
    explicit UnicodeConverter(UErrorCode &errorCode);
    icu::UnicodeString toUnicode(const std::string &s) const;
    icu::UnicodeString process(const std::string &s, UErrorCode &errorCode) const;
  private:
    const Normalizer2 &nfd;
};
#endif
