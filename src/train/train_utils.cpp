#include <string>
#include <tuple>
#include <vector>
#include <iomanip>
#include <iostream>

#include "data.h"
#include "model.h"
#include "state.h"
#include "train_utils.h"
#include "train_loop.h"

void runTraining(const Config &config,
                 const std::string &modelDir, const std::string &dataDir,
                 size_t batchSize, size_t numWorkers, size_t numEpochs) {
  BertModel model(config);
  BinaryClassifier classifier(config);
  loadState(modelDir, *model);

  TextDatasetType trainDataset = readFileToDataset(modelDir, true, dataDir + "/train");
  TextDatasetType devDataset = readFileToDataset(modelDir, true, dataDir + "/dev");

	TextDataLoaderType trainLoader = torch::data::make_data_loader(
    trainDataset,
    torch::data::DataLoaderOptions().batch_size(batchSize).workers(numWorkers));

	TextDataLoaderType devLoader = torch::data::make_data_loader(
    devDataset,
    torch::data::DataLoaderOptions().batch_size(batchSize).workers(numWorkers));

	model->to(torch::kCUDA);
	classifier->to(torch::kCUDA);


  std::vector<torch::Tensor> parameters = model->parameters(true);
  std::vector<torch::Tensor> classifierParameters = classifier->parameters(true);
  parameters.insert(parameters.begin(), classifierParameters.begin(), classifierParameters.end());
  torch::optim::Adam optimizer(parameters, torch::optim::AdamOptions(1e-5));
  torch::nn::BCEWithLogitsLoss criterion;

  std::vector<float> losses;
  std::vector<torch::Tensor> predictions;

  for (int epoch=1; epoch <= numEpochs; epoch++) {
    auto stats = trainLoop(model, classifier, trainLoader, criterion, optimizer);
    losses = std::get<0>(stats);
    predictions = std::get<1>(stats);

    float sum = 0.0f;
    for (float x : losses ) sum += x;
    float avg = sum / losses.size();

    std::cout << "epoch=" << epoch << ", ";
    std::cout << "avg_loss=" << std::fixed << std::setprecision(3) << avg << std::endl;

    stats = trainLoop(model, classifier, devLoader, criterion);
    losses = std::get<0>(stats);
    predictions = std::get<1>(stats);

    sum = 0.0f;
    for (float x : losses ) sum += x;
    avg = sum / losses.size();

    std::cout << "\tval_loss=" << std::fixed << std::setprecision(3) << avg << std::endl;
	}
}
