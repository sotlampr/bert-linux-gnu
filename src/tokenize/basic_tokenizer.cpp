#include <unicode/ustream.h>
#include "basic_tokenizer.h"
#include <unicode/schriter.h>
#include <unicode/brkiter.h>

BasicTokenizer::BasicTokenizer(bool doLowerCase) : doLowerCase (doLowerCase) {};

std::vector<icu::UnicodeString> BasicTokenizer::tokenize(icu::UnicodeString &s) const {
  s = clean(s);
  s = tokenizeCJKChars(s);
  s = s.trim();
  std::vector<icu::UnicodeString> origTokens = whitespaceTokenize(s);
  std::vector<icu::UnicodeString> splitToken, splitTokens;

  for (icu::UnicodeString& token:origTokens) {
    if (token == "[SEP]") {
      splitTokens.push_back(token);
      continue;
    }
    if (doLowerCase) token = token.toLower();
    token = stripAccents(token);
    splitToken = splitPunctuation(token);
    splitTokens.insert(splitTokens.end(), splitToken.begin(), splitToken.end());
  }
  s.remove();
  for (const  icu::UnicodeString& t : splitTokens) {
    s += t;
    s += " ";
  }
  return whitespaceTokenize(s);
};

icu::UnicodeString BasicTokenizer::clean(const icu::UnicodeString &i) const {
  icu::UnicodeString o;
  const UChar *iBuffer = i.getBuffer();
  UCharCharacterIterator it(iBuffer, i.length());
  UChar32 c;
  while (it.hasNext()) {
    c = it.next32PostInc();
    if (c == 0 || c == 0xfffd || u_iscntrl(c)) {
      continue;
    } else if (u_isspace(c)) {
      o += ' ';
    } else {
      o += c;
    }
  }
  return o;
};

icu::UnicodeString BasicTokenizer::tokenizeCJKChars(const icu::UnicodeString &i) const {
  icu::UnicodeString o;
  const UChar *iBuffer = i.getBuffer();
  UCharCharacterIterator it(iBuffer, i.length());
  UChar32 c;
  while (it.hasNext()) {
    c = it.next32PostInc();
    if (   (c >= 0x4e00  && c <= 0x9fff) 
        || (c >= 0x3400  && c <= 0x4dbf) 
        || (c >= 0x20000 && c <= 0x2a6df) 
        || (c >= 0x2a700 && c <= 0x2b73f) 
        || (c >= 0x2b740 && c <= 0x2b81f) 
        || (c >= 0x2b820 && c <= 0x2ceaf) 
        || (c >= 0xf900  && c <= 0xfaff) 
        || (c >= 0x2f800 && c <= 0x2fa1f)) {
      o += " ";
      o += c;
      o += + " ";
    } else {
      o += c;
    }
}
return o;
}

std::vector<icu::UnicodeString> BasicTokenizer::whitespaceTokenize(const icu::UnicodeString &s) const {
  icu::UnicodeString t;
  std::vector<icu::UnicodeString> o;
  const UChar *sBuffer = s.getBuffer();
  UCharCharacterIterator it(sBuffer, s.length());
  UChar32 c;
  while (it.hasNext()) {
    c = it.next32PostInc();
    if (u_isspace(c)) {
      o.push_back(t);
      t.remove();
    } else {
      t += c;
    }
  }
  if (t.length() > 0) {
    o.push_back(t);
  }
  return o;
}

icu::UnicodeString BasicTokenizer::stripAccents(const icu::UnicodeString &i) const {
  icu::UnicodeString o;
  const UChar *iBuffer = i.getBuffer();
  UCharCharacterIterator it(iBuffer, i.length());
  UChar32 c;
  while (it.hasNext()) {
    c = it.next32PostInc();
    if (u_charType(c) == U_NON_SPACING_MARK) {
      continue;
   } else {
      o += c;
    }
  }
  return o;
}

std::vector<icu::UnicodeString> BasicTokenizer::splitPunctuation(const icu::UnicodeString s) const {
  icu::UnicodeString t;
  std::vector<icu::UnicodeString> o;
  const UChar *sBuffer = s.getBuffer();
  UCharCharacterIterator it(sBuffer, s.length());
  UChar32 c;
  while (it.hasNext()) {
    c = it.next32PostInc();
    if (u_ispunct(c)
        || (c >= 33  && c <= 47)
        || (c >= 58  && c <= 64)
        || (c >= 91  && c <= 96)
        || (c >= 123 && c <= 126)) {
      o.push_back(t);
      t.remove();
      t = c;
      o.push_back(t);
      t.remove();
    } else {
      t += c;
    }
  }
  if (t.length() > 0) {
    o.push_back(t);
  }
  return o;
}



