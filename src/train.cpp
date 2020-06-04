#include "data.h"
#include "model.h"
#include "state.h"
#include "tokenize.h"

size_t batchSize = 32;
size_t numWorkers = 4;
size_t epochs = 10;

int main() {
  Config config;
  BertModel model(config);
  BinaryClassifier classifier(config);
  std::cout << model << std::endl;
  loadState("models/bert-base-uncased", *model);

  FullTokenizer *tokenizer = new FullTokenizer("models/bert-base-uncased/vocab.txt", true);

  std::cout << "Reading data...";
  std::vector<std::string> trainTexts = readTextFile("glue/data/CoLA/processed/train-texts.txt");
  std::vector<std::string> devTexts = readTextFile("glue/data/CoLA/processed/dev-texts.txt");
  std::vector<long> trainLabels = readClassificationLabels("glue/data/CoLA/processed/train-labels.txt");
  std::vector<long> devLabels = readClassificationLabels("glue/data/CoLA/processed/dev-labels.txt");
  std::cout << "	OK!" << std::endl;

  std::cout << "Tokenizing...";
  std::vector<std::vector<long>> *trainTextsIds = new std::vector<std::vector<long>>;
  std::vector<std::vector<long>> *devTextsIds = new std::vector<std::vector<long>>;

  for (auto it = trainTexts.begin(); it != trainTexts.end(); it++) {
    trainTextsIds->push_back(tokenizer->tokenizeToIds(*it));
  }

  for (auto it = devTexts.begin(); it != devTexts.end(); it++) {
    devTextsIds->push_back(tokenizer->tokenizeToIds(*it));
  }
  long sosId = tokenizer->tokenToId("[CLS]"), eosId = tokenizer->tokenToId("[SEP]");
  delete tokenizer;
  std::cout << "	OK!" << std::endl;

  std::cout << "Initializing DataLoaders...";
	auto trainDataset = TextDataset(*trainTextsIds, trainLabels, sosId, eosId)
    .map(torch::data::transforms::Stack<>());
	auto devDataset = TextDataset(*devTextsIds, devLabels, sosId, eosId);

  delete trainTextsIds;
  delete devTextsIds;

	auto trainLoader = torch::data::make_data_loader(
    trainDataset,
    torch::data::DataLoaderOptions().batch_size(batchSize).workers(numWorkers));

	auto devLoader = torch::data::make_data_loader(
    devDataset, torch::data::DataLoaderOptions().batch_size(batchSize));
  std::cout << "	OK!" << std::endl;

  std::cout << "Moving model to CUDA...";
	model->to(torch::kCUDA);
	classifier->to(torch::kCUDA);
  std::cout << "	OK!" << std::endl;

  std::vector<torch::Tensor> parameters = model->parameters(true);
  std::vector<torch::Tensor> classifierParameters = classifier->parameters(true);
  parameters.insert(parameters.begin(), classifierParameters.begin(), classifierParameters.end());
  torch::optim::Adam optimizer(parameters, torch::optim::AdamOptions(1e-4));
  torch::nn::BCEWithLogitsLoss criterion;


  for (int epoch=1; epoch <= epochs; epoch++) {
    int step = 1;
    for (auto& batch : *trainLoader) {
        auto data = batch.data.cuda();
        auto labels = batch.target.cuda().to(torch::kFloat);

        torch::Tensor output = model->forward(data);
        torch::Tensor logits = classifier->forward(output).squeeze();

        torch::Tensor loss = criterion->forward(logits, labels);
        if (step % 50 == 0) {
          std::cout << "epoch=" << epoch << ", step=" << step << ", loss=" << loss.item<float>() << std::endl;
        }
        model->zero_grad();
        loss.backward();
        optimizer.step();
        step++;
    }
	}
  return 0;
}
