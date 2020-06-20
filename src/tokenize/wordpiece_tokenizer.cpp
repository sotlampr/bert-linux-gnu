#include "wordpiece_tokenizer.h"
#include <unicode/schriter.h>

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
  int32_t start = 0;
  bool isBad = false;

  // Greedy longest-match first using the given vocabulary
  while (start < s.length()) {
    int32_t end = s.length();
    curSubString.remove(); // Reset `curSubString`
    while (start < end) {
      s.extract(start, end-start, subString);
      if (start > 0) {
        // Not word boundary, insert '##' before
        subString = subString.insert(0, "##");
      }
      if (vocab.find(subString) != vocab.end()) {
        // Token is found in the vocabulary
        curSubString = subString;
        break;
      }
      end -= 1;
    }
    if (curSubString.length() == 0) {
      // No match, to be tokenized as [UNK]
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
