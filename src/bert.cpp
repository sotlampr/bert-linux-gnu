#include <iostream>
#include <string>

#include "test.h"
#include "train.h"
#include "tokenize.h"

void printHelp(const std::string& programName) {
    std::cout << "Usage: "
              << programName
              << " {train,tokenize} [OPTIONS...]"
              << std::endl;
}


int main(int argc, char* argv[]) {
  if (argc < 2) {
    printHelp(argv[0]);
    return 1;
  }

  std::string a1 = argv[1];
  if (a1 == "-h" || a1 == "--help") {
    printHelp(argv[0]);
    return 1;
  }

  if (std::string(argv[1]) == "train") {
    return train::main(argc-1,  ++argv);
  } else if (std::string(argv[1]) == "tokenize") {
    return tokenize::main(argc-1,  ++argv);
  } else if (std::string(argv[1]) == "test") {
    return test::main(argc-1,  ++argv);
  } else {
    std::cout << "Invalid command `" << argv[1] << "`" << std::endl;
    return 1;
  }
  return 0;
}

