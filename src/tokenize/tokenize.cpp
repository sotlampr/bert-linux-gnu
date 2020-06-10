#include "tokenize.h"

#include <fstream>
#include <iostream>
#include <vector>

#include "full_tokenizer.h"

namespace tokenize {

int main(int argc, char *argv[]) {
	if (argc != 3) {
			std::cout << "Usage: " << argv[0] << " [MODEL_DIR] [FILE]" << std::endl;
			return 1;
	}

  std::string modelDir = argv[1];

  FullTokenizer *tokenizer = new FullTokenizer(modelDir);
  std::ifstream file(argv[2]);
  std::string line;
  while (std::getline(file, line)) {
    auto tokens = tokenizer->tokenize(line);
    for (size_t i = 0; i < tokens.size() - 1; i++) {
      std::cout << tokens[i] << " ";
    }
    std::cout << tokens.back() << std::endl;
  }
  file.close();
  return 0;
}

}
