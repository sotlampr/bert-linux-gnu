#include "predict.h"

#include <string>
#include <vector>
#include <glob.h>

#include <torch/nn/modules/container/any.h>
#include <torch/serialize.h>
#include <torch/types.h>

#include "config.h"
#include "data.h"
#include "model/bert_model.h"
#include "model/classifier.h"
#include "state.h"


namespace predict {

int main(int argc, char *argv[]) {
  if ((argc != 3) && (argc != 4)) {
      std::cout << "Usage: " << argv[0] << " MODEL FILE [TASK]" << std::endl;
      return 1;
  }

  std::string baseFname = argv[1];
  std::string textsFname = argv[2];
  std::string taskName;
  if (argc == 4) {
    taskName = argv[3];
  } else {
    taskName = "";
  }
  std::string bertFname = baseFname + "-bert.pt";
  torch::nn::AnyModule classifier;

  Config config;
  readStruct(config, baseFname + "-bert.config");
  BertModel bertModel(config);
  torch::load(bertModel, bertFname);

  for (const auto& fname : getGlobFiles(baseFname + "*" + taskName + "*.pt")) {
    torch::nn::AnyModule module;
    if (fname.find("binary") != std::string::npos) {
      BinaryClassifierOptions options;
      std::string optionsFname = fname;
      std::string::size_type i = optionsFname.rfind('.', optionsFname.length());
      optionsFname.replace(i+1, 6, "config");
      readStruct(options, optionsFname);
      BinaryClassifier clf(options);
      torch::load(clf, fname);
      classifier = torch::nn::AnyModule(clf);
    } else if (fname.find("multiclass") != std::string::npos) {
      MutliclassClassifierOptions options;
      std::string optionsFname = fname;
      std::string::size_type i = optionsFname.rfind('.', optionsFname.length());
      optionsFname.replace(i+1, 6, "config");
      readStruct(options, optionsFname);
      MulticlassClassifier clf(options);
      torch::load(clf, fname);
      classifier = torch::nn::AnyModule(clf);
    }
  }

  std::string vocabFname = baseFname + ".vocab";
  std::string lowercaseFname = baseFname + ".lowercase";
  torch::Tensor texts = readTextsToTensor(textsFname, vocabFname, lowercaseFname);
  bertModel->eval();
  classifier.ptr()->eval();

  for (int i = 0; i < texts.size(0); i++) {
    torch::Tensor hidden = bertModel->forward(texts.index({i}).unsqueeze(0).cuda());
    torch::Tensor logits = classifier.forward(hidden).squeeze(0);
    std::cout << logits.item<float>() << std::endl;
  }

  // Get the first prediction to deduct output shape
  // torch::Tensor hidden = bertModel->forward(texts.index({0}).unsqueeze(0).cuda());
  // torch::Tensor logits = classifier.forward(hidden).squeeze(0);
  // std::cout << "hidden sizes: " << hidden.sizes() << std::endl;
  // std::cout << "logits sizes: " << logits.sizes() << std::endl;
  // std::cout << "logits: " << logits << std::endl;

  // std::vector<long> outputShape = logits.sizes().vec();
  // outputShape.insert(outputShape.begin(), texts.size(0));

  // torch::Tensor outputs = torch::empty(outputShape);
  // std::cout << "outputs size: " << outputs.sizes() << std::endl;

  // for (size_t i = 0; i < texts.size(0); i++) {
  // }

  return 0;

}
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
