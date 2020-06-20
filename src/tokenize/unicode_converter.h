#ifndef UNICODE_CONVERTER_H
#define UNICODE_CONVERTER_H
#include <string>

#include <unicode/ustream.h>
#include <unicode/normalizer2.h>

class UnicodeConverter {
  public:
    explicit UnicodeConverter(UErrorCode &errorCode);
    // Convert to unicode and NFD normalize an std::string
    icu::UnicodeString process(const std::string &s, UErrorCode &errorCode) const;
  private:
    // Convert an std::string to icu::UnicodeString
    icu::UnicodeString toUnicode(const std::string &s) const;
    const Normalizer2 &nfd;
};
#endif
