#include <fstream>
#include <iostream>
#include <cassert>
#include <map>
#include <vector>
#include <iterator>
#include <stdexcept>

#include <unicode/ustream.h>
#include <unicode/schriter.h>
#include <unicode/brkiter.h>
#include <unicode/normalizer2.h>

#include "tokenize.h"

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

BasicTokenizer::BasicTokenizer(bool doLowerCase) : doLowerCase (doLowerCase) {};

std::vector<icu::UnicodeString> BasicTokenizer::tokenize(icu::UnicodeString &s) const {
  s = clean(s);
  s = tokenizeCJKChars(s);
  s = s.trim();
  std::vector<icu::UnicodeString> origTokens = whitespaceTokenize(s);
  std::vector<icu::UnicodeString> splitToken, splitTokens;

  for (icu::UnicodeString& token:origTokens) {
    if (doLowerCase) {
      token = token.toLower();
    }
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
  UCharCharacterIterator it(iBuffer, u_strlen(iBuffer));
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
  UCharCharacterIterator it(iBuffer, u_strlen(iBuffer));
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
  UCharCharacterIterator it(sBuffer, u_strlen(sBuffer));
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
  UCharCharacterIterator it(iBuffer, u_strlen(iBuffer));
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
  UCharCharacterIterator it(sBuffer, u_strlen(sBuffer));
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


WordPieceTokenizer::WordPieceTokenizer(const std::map<icu::UnicodeString,long> &vocab,
                                       const icu::UnicodeString &unkToken,
                                       int maxInputCharsPerWord)
  : vocab (vocab), unkToken (unkToken), maxInputCharsPerWord(maxInputCharsPerWord) { }

std::vector<icu::UnicodeString> WordPieceTokenizer::tokenize(const icu::UnicodeString &s) const {
  std::vector<icu::UnicodeString> out;
  if (s.length() > maxInputCharsPerWord) {
    out.push_back(unkToken);
    return out;
  }

  icu::UnicodeString subString, curSubString;
  const UChar *sBuffer = s.getBuffer();
  UCharCharacterIterator it(sBuffer, u_strlen(sBuffer));
  int32_t start = 0;
  bool isBad = false;

  while (start < s.length()) {
    int32_t end = s.length();
    curSubString.remove();
    while (start < end) {
      s.extract(start, end-start, subString);
      if (start > 0) {
        subString = subString.insert(0, "##");
      }
      if (vocab.find(subString) != vocab.end()) {
        curSubString = subString;
        break;
      }
      end -= 1;
    }
    if (curSubString.length() == 0) {
      isBad = true;
      break;
    }
    out.push_back(curSubString);
    start = end;
  }

  if (isBad) {
    out.push_back(unkToken);
  }
  return out;
}

std::vector<long> WordPieceTokenizer::tokensToIds(const std::vector<icu::UnicodeString> &v) const {
  std::vector<long> ids;
  for (auto it = v.begin(); it != v.end(); it++) {
    ids.push_back(vocab.at(*it));
  }
  return ids;
};

long WordPieceTokenizer::tokenToId(const icu::UnicodeString &s) const {
  return vocab.at(s);
};

FullTokenizer::FullTokenizer(const std::string &vocabFile, bool doLowerCase)
  : doLowerCase (doLowerCase),
    unicoder (*(new UnicodeConverter(uErr))),
    basicTokenizer (*(new BasicTokenizer(doLowerCase))),
    wordPieceTokenizer (*(new WordPieceTokenizer(readVocabulary(vocabFile).first, "[UNK]", 200))) {};

FullTokenizer::~FullTokenizer() {
  delete &unicoder;
  delete &basicTokenizer;
  delete &wordPieceTokenizer;
};

std::pair<std::map<icu::UnicodeString,long>,std::map<long,icu::UnicodeString>>
FullTokenizer::readVocabulary(const std::string &vocabFile) {
  std::ifstream file(vocabFile);
  if (!file.is_open()) {
    throw std::runtime_error(vocabFile + " not found!");
  }
  std::string line;
  icu::UnicodeString uLine;
  long i = 0;
  while (std::getline(file, line)) {
    uLine = unicoder.process(line, uErr);
    vocab.insert({uLine, i});
    invVocab.insert({i, uLine});
    i++;
  }
  assert (!vocab.empty());
  return std::make_pair(vocab, invVocab);
}

std::vector<icu::UnicodeString> FullTokenizer::tokenize(const std::string &s) {
  icu::UnicodeString us = unicoder.process(s, uErr);
  std::vector<icu::UnicodeString> o;
  std::vector<icu::UnicodeString> wordPieces;
  for (icu::UnicodeString& token : basicTokenizer.tokenize(us)){
    wordPieces = wordPieceTokenizer.tokenize(token);
    o.insert(o.end(), wordPieces.begin(), wordPieces.end());
  }
  return o;
}

std::vector<long> FullTokenizer::tokenizeToIds (const std::string &s) {
  return wordPieceTokenizer.tokensToIds(tokenize(s));
}

long FullTokenizer::tokenToId(const icu::UnicodeString &s) const {
  return wordPieceTokenizer.tokenToId(s);
}
