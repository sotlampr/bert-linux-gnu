#include "unicode_converter.h"

UnicodeConverter::UnicodeConverter(UErrorCode &errorCode)
      : nfd (*Normalizer2::getNFDInstance(errorCode)) {}

UnicodeString UnicodeConverter::toUnicode(const std::string &s) const {
  return icu::UnicodeString::fromUTF8(StringPiece(s.c_str()));
}

UnicodeString UnicodeConverter::process(const std::string &s, UErrorCode &errorCode) const {
  UnicodeString us = toUnicode(s);
  if (!U_SUCCESS(errorCode)) {
    throw std::runtime_error("Unicode conversion failed");
  }
  us = nfd.normalize(us, errorCode);
  if (!U_SUCCESS(errorCode)) {
    throw std::runtime_error("Unicode normalization failed");
  }
  return us;
}
