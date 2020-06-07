#include <string>
#include <tuple>
#include <vector>
#include <iomanip>
#include <iostream>

#include "data.h"
#include "model.h"
#include "state.h"
#include "metrics.h"
#include "train_utils.h"
#include "train_loop.h"

void runTraining(const Config &config,
                 const std::string &modelDir, const std::string &dataDir,
                 size_t batchSize, size_t numWorkers, size_t numEpochs) {
  BertModel model(config);
  BinaryClassifier classifier(config);
  loadState(modelDir, *model);

  TextDatasetType trainDataset = readFileToDataset(modelDir, true, dataDir + "/train");
  TextDatasetType valDataset = readFileToDataset(modelDir, true, dataDir + "/dev");

  auto trainLabelsSize  = trainDataset.dataset().getLabels().sizes();
  auto valLabelsSize  = valDataset.dataset().getLabels().sizes();

	TextDataLoaderType trainLoader = torch::data::make_data_loader(
    trainDataset,
    torch::data::DataLoaderOptions().batch_size(batchSize).workers(numWorkers));

	TextDataLoaderType valLoader = torch::data::make_data_loader(
      valDataset,
      torch::data::DataLoaderOptions().batch_size(batchSize).workers(numWorkers));

	model->to(torch::kCUDA);
	classifier->to(torch::kCUDA);


  std::vector<torch::Tensor> parameters = model->parameters(true);
  std::vector<torch::Tensor> classifierParameters = classifier->parameters(true);
  parameters.insert(parameters.begin(), classifierParameters.begin(), classifierParameters.end());
  torch::optim::Adam optimizer(parameters, torch::optim::AdamOptions(1e-5));
  torch::nn::BCEWithLogitsLoss criterion;

  for (int epoch=1; epoch <= numEpochs; epoch++) {
    std::vector<float> trainLosses, valLosses;
    torch::Tensor trainLabels = torch::zeros(trainLabelsSize);
    torch::Tensor trainPredictions = torch::zeros(trainLabelsSize);
    torch::Tensor valLabels = torch::zeros(valLabelsSize);
    torch::Tensor valPredictions = torch::zeros(valLabelsSize);

    trainLoop(model, classifier, trainLoader, criterion, optimizer, trainLosses, trainLabels, trainPredictions);

    float sum = 0.0f;
    for (float x : trainLosses ) sum += x;
    float avg = sum / trainLosses.size();
    trainPredictions = (trainPredictions >= 0.0f).to(torch::kInt64);
    float mcc = matthewsCorrelationCoefficient(trainLabels, trainPredictions);

    std::cout << "epoch=" << epoch << ", ";
    std::cout << "avg_loss=" << std::fixed << std::setprecision(3) << avg;
    std::cout << " mcc=" << std::fixed << std::setprecision(3) << mcc << std::endl;

    trainLoop(model, classifier, valLoader, criterion, valLosses, valLabels, valPredictions);

    sum = 0.0f;
    for (float x : valLosses ) sum += x;
    avg = sum / valLosses.size();

    valPredictions = (valPredictions >= 0.0f).to(torch::kInt64);
    mcc = matthewsCorrelationCoefficient(valLabels, valPredictions);

    std::cout << "\tval_loss=" << std::fixed << std::setprecision(3) << avg;
    std::cout << " val_mcc=" << std::fixed << std::setprecision(3) << mcc << std::endl;
	}
}
