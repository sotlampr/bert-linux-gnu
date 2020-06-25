#include "sentencepiece_tokenizer.h"

#include <algorithm>
#include <stdexcept>
#include <string>

SentencepieceTokenizer::SentencepieceTokenizer(const std::string& modelFname,
                                               const std::string& lowercaseFname)
    : sentencepieceProcessor (*(new sentencepiece::SentencePieceProcessor())),
      doLowerCase (getDoLowercase(lowercaseFname))  {
  const auto status = sentencepieceProcessor.Load(modelFname);
  if (!status.ok()) {
    throw std::runtime_error("Could not load sentencepiece");
	}
}

bool SentencepieceTokenizer::getDoLowercase(const std::string& lowercaseFname) {
  std::ifstream file(lowercaseFname);
  // If there exists file named `lowercase`, do lowercase
  if (file.is_open()) return true;
  return false;
}


SentencepieceTokenizer::~SentencepieceTokenizer() {
  delete &sentencepieceProcessor;
};

std::string SentencepieceTokenizer::handleCase(std::string s) const {
  if (doLowerCase) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c){ return std::tolower(c); });
  }
  return s;
}

std::vector<std::string> SentencepieceTokenizer::tokenize(const std::string &s) {
	return sentencepieceProcessor.EncodeAsPieces(handleCase(s));
}

std::vector<long> SentencepieceTokenizer::tokenizeToIds (const std::string &s) {
  std::vector<int> intIds = sentencepieceProcessor.EncodeAsIds(handleCase(s));
  std::vector<long> out(intIds.begin(), intIds.end());
  return out;
}

long SentencepieceTokenizer::tokenToId(const std::string &s) const {
	return static_cast<long>(sentencepieceProcessor.PieceToId(s));
}
