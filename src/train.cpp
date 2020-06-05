#include "data.h"
#include "model.h"
#include "state.h"
#include "data.h"

size_t batchSize = 32;
size_t numWorkers = 4;
size_t epochs = 10;


int main() {
  Config config;
  BertModel model(config);
  BinaryClassifier classifier(config);
  loadState("models/bert-base-uncased", *model);

  auto trainDataset = readFileToDataset("models/bert-base-uncased", true,
                                        "glue/data/CoLA/processed/train");
  auto devDataset = readFileToDataset("models/bert-base-uncased", true,
                                      "glue/data/CoLA/processed/dev");

  std::cout << "trainDataset size=" << trainDataset.size() << std::endl;

  std::cout << "Initializing DataLoaders...";

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


  std::cout << "Initializing optimizer+loss...";
  std::vector<torch::Tensor> parameters = model->parameters(true);
  std::vector<torch::Tensor> classifierParameters = classifier->parameters(true);
  parameters.insert(parameters.begin(), classifierParameters.begin(), classifierParameters.end());
  torch::optim::Adam optimizer(parameters, torch::optim::AdamOptions(1e-4));
  torch::nn::BCEWithLogitsLoss criterion;
  std::cout << "	OK!" << std::endl;


  std::cout << "Starting training..." << std::endl;
  for (int epoch=1; epoch <= epochs; epoch++) {
    int step = 1;
    for (auto& batch : *trainLoader) {
        std::cout << "epoch=" << epoch << ", step=" << step << std::endl;
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
