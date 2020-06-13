#include "test.h"

#include <string>
#include <vector>
#include <glob.h>

#include <torch/nn/modules/container/any.h>
#include <torch/serialize.h>
#include <torch/types.h>

#include "config.h"
#include "model/bert_model.h"
#include "model/classifier.h"
#include "state.h"


namespace test {

int main(int argc, char *argv[]) {
  if (argc != 3) {
      std::cout << "Usage: " << argv[0] << " [MODEL] [FILE]" << std::endl;
      return 1;
  }

  std::string baseFname = argv[1];
  std::string bertFname = baseFname + "-bert.pt";
  std::vector<torch::nn::AnyModule> classifiers;

  Config config;
  readStruct(config, baseFname + "-bert.config");
  BertModel bertModel(config);
  torch::load(bertModel, bertFname);

  for (const auto& fname : getGlobFiles(baseFname + "-*.pt")) {
    torch::nn::AnyModule module;
    if (fname.find("binary") != std::string::npos) {
      BinaryClassifierOptions options;
      std::string optionsFname = fname;
      std::string::size_type i = optionsFname.rfind('.', optionsFname.length());
      optionsFname.replace(i+1, 6, "config");
      readStruct(options, optionsFname);
      BinaryClassifier classifier(options);
      torch::load(classifier, fname);
      classifiers.push_back(torch::nn::AnyModule(classifier));
    } else if (fname.find("multiclass") != std::string::npos) {
      MutliclassClassifierOptions options;
      std::string optionsFname = fname;
      std::string::size_type i = optionsFname.rfind('.', optionsFname.length());
      optionsFname.replace(i+1, 6, "config");
      readStruct(options, optionsFname);
      MulticlassClassifier classifier(options);
      torch::load(classifier, fname);
      classifiers.push_back(torch::nn::AnyModule(classifier));
    }
  }

  torch::Tensor inputs = torch::zeros({1, 100}).cuda().to(torch::kInt64);

  std::cout << "BertModel->forward...";
  torch::Tensor hidden = bertModel->forward(inputs);
  std::cout << "\tOK" << std::endl;

  torch::Tensor clfOut;
  for (auto& classifier : classifiers) {
    std::cout << classifier.ptr()->name() << "->forward...";
    clfOut = classifier.forward(hidden);
    std::cout << "\tOK" << std::endl;
  }
  return 0;

  // std::string modelDir = argv[1];

  // FullTokenizer *tokenizer = new FullTokenizer(modelDir);
  // std::ifstream file(argv[2]);
  // std::string line;
  // while (std::getline(file, line)) {
  //   auto tokens = tokenizer->tokenize(line);
  //   for (size_t i = 0; i < tokens.size() - 1; i++) {
  //     std::cout << tokens[i] << " ";
  //   }
  //   std::cout << tokens.back() << std::endl;
  // }
  // file.close();
  // return 0;
}

}
