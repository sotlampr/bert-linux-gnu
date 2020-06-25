#include "wordpiece_tokenizer.h"
#include <unicode/schriter.h>

WordPieceTokenizer::WordPieceTokenizer(const std::map<std::string, long> &vocab,
                                       const std::string &unkToken,
                                       size_t maxInputCharsPerWord)
  : vocab (vocab), unkToken (unkToken), maxInputCharsPerWord(maxInputCharsPerWord) { }

std::vector<std::string> WordPieceTokenizer::tokenize(const std::string &s) const {
  std::vector<std::string> out;
  if (s.length() > maxInputCharsPerWord) {
    out.push_back(unkToken);
    return out;
  }

  std::string subString, curSubString;
  size_t start = 0;
  bool isBad = false;

  // Greedy longest-match first using the given vocabulary
  while (start < s.length()) {
    size_t end = s.length();
    curSubString.clear(); // Reset `curSubString`
    while (start < end) {
      subString = s.substr(start, end-start);
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

std::vector<long> WordPieceTokenizer::tokensToIds(const std::vector<std::string> &v) const {
  std::vector<long> ids;
  for (auto it = v.begin(); it != v.end(); it++) {
    ids.push_back(vocab.at(*it));
  }
  return ids;
};

long WordPieceTokenizer::tokenToId(const std::string &s) const {
  return vocab.at(s);
};
