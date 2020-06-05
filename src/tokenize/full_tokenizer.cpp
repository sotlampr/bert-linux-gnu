#include "full_tokenizer.h"
#include <fstream>
#include <map>
#include <vector>

#include <unicode/ustream.h>


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
    throw std::runtime_error(vocabFile + " not found");
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
  if (vocab.empty()) {
    throw std::runtime_error("Vocabulary is empty");
  }
  return std::make_pair(vocab, invVocab);
}

std::vector<icu::UnicodeString> FullTokenizer::tokenize(const std::string &s) {
  icu::UnicodeString us = unicoder.process(s, uErr);
  std::vector<icu::UnicodeString> o;
  std::vector<icu::UnicodeString> wordPieces;
  #ifdef DEBUG
  std::cout << "Tokenizing `" << s << std::endl;
  #endif
  for (icu::UnicodeString& token : basicTokenizer.tokenize(us)){
    wordPieces = wordPieceTokenizer.tokenize(token);
    o.insert(o.end(), wordPieces.begin(), wordPieces.end());
    #ifdef DEBUG
      std::cout << "\t`" << token << "` ->";
      for (const auto t : wordPieces) {
        std::cout << "`" << t << "`,";
      }
      std::cout << std::endl;
    #endif
  }
  return o;
}

std::vector<long> FullTokenizer::tokenizeToIds (const std::string &s) {
  return wordPieceTokenizer.tokensToIds(tokenize(s));
}

long FullTokenizer::tokenToId(const icu::UnicodeString &s) const {
  return wordPieceTokenizer.tokenToId(s);
}
