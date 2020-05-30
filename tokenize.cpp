#include <fstream>
#include <iostream>
#include <cassert>
#include <map>
#include <vector>
#include <iterator>

#include <unicode/ustream.h>
#include <unicode/schriter.h>
#include <unicode/brkiter.h>
#include <unicode/normalizer2.h>

// Convert to icu::UnicodeString and perform 'NFD' normalization
class UnicodeConverter {
  public:
    UnicodeConverter(UErrorCode &errorCode)
      : nfd (*Normalizer2::getNFDInstance(errorCode)) {}

    UnicodeString toNKD(const UnicodeString s, UErrorCode &errorCode) {
      return nfd.normalize(s, errorCode);
    }

    UnicodeString toUnicode(const std::string s) {
      return icu::UnicodeString::fromUTF8(StringPiece(s.c_str()));
    }

    UnicodeString process(const std::string s, UErrorCode &errorCode) {
      UnicodeString us = toUnicode(s);
      return nfd.normalize(us, errorCode);
    }
  private:
    const Normalizer2 &nfd;
};

// Basic tokenizer: Splits punctuation, CJK chars and whitecpace.
class BasicTokenizer {
  public:
    BasicTokenizer(bool doLowerCase) : doLowerCase (doLowerCase) {};

    std::vector<icu::UnicodeString> tokenize(UnicodeString s) const {
      std::vector<icu::UnicodeString> output_tokens;
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

  private:
    const bool doLowerCase;

    // Invalid character removal and whitespace cleanup
    icu::UnicodeString clean(icu::UnicodeString &i) const {
      icu::UnicodeString o;
      const UChar *iBuffer = i.getTerminatedBuffer();
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

    // Add whitespace between CJK characters
    icu::UnicodeString tokenizeCJKChars(icu::UnicodeString &i) const {
      icu::UnicodeString o;
      const UChar *iBuffer = i.getTerminatedBuffer();
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

  // Split by whitespace
  std::vector<icu::UnicodeString> whitespaceTokenize(icu::UnicodeString s) const {
      icu::UnicodeString t;
      std::vector<icu::UnicodeString> o;
      const UChar *sBuffer = s.getTerminatedBuffer();
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

  //  Strip 'Nm' category unicode accents
  icu::UnicodeString stripAccents(icu::UnicodeString &i) const {
      icu::UnicodeString o;
      const UChar *iBuffer = i.getTerminatedBuffer();
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

  // Split at punctuation 
  std::vector<icu::UnicodeString> splitPunctuation(icu::UnicodeString s) const {
      icu::UnicodeString t;
      std::vector<icu::UnicodeString> o;
      const UChar *sBuffer = s.getTerminatedBuffer();
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
};


// WordPiece tokenizer: longest-match-first tokenization given a vocabulary
class WordPieceTokenizer {
  public:
    WordPieceTokenizer(std::map<icu::UnicodeString,int> vocab,
                       icu::UnicodeString unkToken,
                       int maxInputCharsPerWord)
      : vocab (vocab), unkToken (unkToken), maxInputCharsPerWord(maxInputCharsPerWord) {
    }

    std::vector<icu::UnicodeString> tokenize(icu::UnicodeString s) const {
      std::vector<icu::UnicodeString> out;
      if (s.length() > maxInputCharsPerWord) {
        out.push_back(unkToken);
        return out;
      }

      icu::UnicodeString subString, curSubString;
      const UChar *sBuffer = s.getTerminatedBuffer();
      UCharCharacterIterator it(sBuffer, u_strlen(sBuffer));
      int32_t start = 0;
      int32_t end;
      bool isBad = false;

      while (start < s.length()) {
        end = s.length();
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

  private:
    std::map<UnicodeString, int> vocab;
    const icu::UnicodeString unkToken;
    const int maxInputCharsPerWord;
};

// Basic- and WordPiece tokenization (also converts to unicode and normalizes)
class FullTokenizer {
  public:
    FullTokenizer(std::string vocabFile, bool doLowerCase)
      : doLowerCase (doLowerCase),
        unicoder (*(new UnicodeConverter(uErr))),
        basicTokenizer (*(new BasicTokenizer(doLowerCase))),
        wordPieceTokenizer (*(new WordPieceTokenizer(readVocabulary(vocabFile).first, "[UNK]", 200))) {};

    std::pair<std::map<icu::UnicodeString,int>,std::map<int,icu::UnicodeString>>
    readVocabulary(std::string vocabFile) {
      std::ifstream file(vocabFile);
      std::string line;
      icu::UnicodeString uLine;
      int i = 0;
      while (std::getline(file, line)) {
        uLine = unicoder.process(line, uErr);
        assert (U_SUCCESS(uErr));
        vocab.insert({uLine, i});
        invVocab.insert({i, uLine});
        i++;
      }
      assert (!vocab.empty());
      return std::make_pair(vocab, invVocab);
    }

    std::vector<icu::UnicodeString> tokenize(std::string s) {
      icu::UnicodeString us = unicoder.process(s, uErr);
      std::vector<icu::UnicodeString> o;
      std::vector<icu::UnicodeString> wordPieces;
      for (icu::UnicodeString& token : basicTokenizer.tokenize(us)){
        wordPieces = wordPieceTokenizer.tokenize(token);
        o.insert(o.end(), wordPieces.begin(), wordPieces.end());
      }

      std::copy(o.begin(), o.end()-1, std::ostream_iterator<icu::UnicodeString>(std::cout, " "));
      std::copy(o.end()-1, o.end(), std::ostream_iterator<icu::UnicodeString>(std::cout));
      std::cout << std::endl;
      return o;
    }

  private:
    std::map<UnicodeString, int> vocab;
    std::map<int, UnicodeString> invVocab;
    UErrorCode uErr;
    const bool doLowerCase;
    UnicodeConverter &unicoder;
    const BasicTokenizer &basicTokenizer;
    WordPieceTokenizer &wordPieceTokenizer;
};

int main(int argc, char *argv[]) {
	if (argc != 2) {
			std::cerr << "Usage: " << argv[0] << " [FILE]" << std::endl;
			return 1;
	}
  FullTokenizer *tokenizer = new FullTokenizer("vocab.txt", true);
  std::ifstream file(argv[1]);
  std::string line;

  while (std::getline(file, line)) {
    tokenizer->tokenize(line);
  }
  file.close();
  return 0;
}
