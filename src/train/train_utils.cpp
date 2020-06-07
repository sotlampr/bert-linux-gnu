#include <string>
#include <tuple>
#include <vector>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "data.h"
#include "model.h"
#include "state.h"
#include "metrics.h"
#include "train_utils.h"
#include "train_loop.h"
#include "task.h"

void initCriteria(std::vector<Task>& tasks) {
  std::for_each(tasks.begin(), tasks.end(),
    [](Task &t) -> Task {
        auto criterion = torch::nn::BCEWithLogitsLoss();
        t.setCriterion<torch::nn::BCEWithLogitsLoss>(std::move(criterion));
        return t;
   });
}

void runTraining(const Config &config,
                 const std::string &modelDir,
                 const std::string &dataDir,
                 std::vector<Task> &tasks,
                 size_t batchSize,
                 size_t numWorkers,
                 size_t numEpochs) {
	// Initialize models
  BertModel model(config);
  BinaryClassifier classifier(config);
  loadState(modelDir, *model);
	model->to(torch::kCUDA);
	classifier->to(torch::kCUDA);

	//Initialize dataset
  TextDatasetType trainDataset = readDataset(modelDir, true, dataDir + "/train", tasks);
  TextDatasetType valDataset = readDataset(modelDir, true, dataDir + "/dev", tasks);

	// Get label tensor sizes
  std::vector<torch::IntArrayRef> trainLabelSizes = trainDataset.dataset().getLabelSizes();
  std::vector<torch::IntArrayRef> valLabelSizes = valDataset.dataset().getLabelSizes();

	// Initialize data loaders
	TextDataLoaderType trainLoader = torch::data::make_data_loader(
    trainDataset,
    torch::data::DataLoaderOptions().batch_size(batchSize).workers(numWorkers));

	TextDataLoaderType valLoader = torch::data::make_data_loader(
      valDataset,
      torch::data::DataLoaderOptions().batch_size(batchSize).workers(numWorkers));

  initCriteria(tasks);

	// Initialize optimizer
  std::vector<torch::Tensor> parameters = model->parameters(true);
  std::vector<torch::Tensor> classifierParameters = classifier->parameters(true);
  parameters.insert(parameters.begin(), classifierParameters.begin(), classifierParameters.end());
  torch::optim::Adam optimizer(parameters, torch::optim::AdamOptions(1e-5));

  for (int epoch=1; epoch <= numEpochs; epoch++) {
    std::cout << "Epoch " << epoch << "..." << std::flush;
    std::vector<std::vector<float>> trainLosses(tasks.size()), valLosses(tasks.size());

    std::vector<torch::Tensor> trainLabels, trainPredictions, valLabels, valPredictions;

    for (auto it = trainLabelSizes.begin();
              it != trainLabelSizes.end();
              it++) {
      trainLabels.push_back(torch::zeros(*it));
      trainPredictions.push_back(torch::zeros(*it));
    }

    for (auto it = valLabelSizes.begin();
              it != valLabelSizes.end();
              it++) {
      valLabels.push_back(torch::zeros(*it));
      valPredictions.push_back(torch::zeros(*it));
    }

    trainLoop(model, classifier, trainLoader, tasks, trainLosses, trainLabels, trainPredictions, optimizer);
    std::cout << "\tOK" << std::endl;
  
    for (size_t i = 0; i < tasks.size(); i++){
      std::cout << "Task: " << tasks[i].getName() << std::endl;
      float sum = 0.0f;
      for (float x : trainLosses[i] ) sum += x;
      float avg = sum / trainLosses[i].size();
      std::cout << "\tavg. loss=" << std::fixed << std::setprecision(3) << avg;
      trainPredictions[i] = (trainPredictions[i] >= 0.0f).to(torch::kInt64);
      float mcc = matthewsCorrelationCoefficient(trainLabels[i], trainPredictions[i]);
      std::cout << " mcc=" << std::fixed << std::setprecision(3) << mcc << std::endl;
    }


    // trainLoop(model, classifier, valLoader, criterion, valLosses, valLabels, valPredictions);

    // sum = 0.0f;
    // for (float x : valLosses ) sum += x;
    // avg = sum / valLosses.size();

    // valPredictions = (valPredictions >= 0.0f).to(torch::kInt64);
    // mcc = matthewsCorrelationCoefficient(valLabels, valPredictions);

    // std::cout << "\tval_loss=" << std::fixed << std::setprecision(3) << avg;
    // std::cout << " val_mcc=" << std::fixed << std::setprecision(3) << mcc << std::endl;
	}
}
