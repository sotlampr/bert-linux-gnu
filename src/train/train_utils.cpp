#include "train_utils.h"

#include <string>
#include <tuple>
#include <vector>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "model.h"
#include "state.h"
#include "metrics.h"
#include "train_loop.h"

std::vector<Task> initTasks(std::vector<Task>& tasks,
                            TextDatasetType& dataset,
                            const Config& config) {
  std::vector<Task> out;
  std::vector<torch::Tensor> weights = dataset.dataset().getClassWeights(tasks);

  for (size_t i = 0; i< tasks.size(); i++) {
    if ((Regression & tasks[i].taskType) == Regression) {
      throw std::runtime_error("Regression not implemented");
    } else if ((Binary & tasks[i].taskType) == Binary) {
      torch::Tensor pos_weight = weights[i];
      out.push_back(
        Task(
          tasks[i],
          BinaryClassifier(config),
          torch::nn::BCEWithLogitsLoss(
            torch::nn::BCEWithLogitsLossOptions().pos_weight(pos_weight)
          ),
          [] (torch::Tensor& logits) -> torch::Tensor {
            return (logits >= 0.0f).to(torch::kInt64);
          }
        )
      );
      out.back().classifier.ptr()->to(torch::kCUDA);
    } else {
      torch::Tensor weight = weights[i];
      out.push_back(
        Task(
          tasks[i],
          MulticlassClassifier(config, weight.size(0)),
          torch::nn::CrossEntropyLoss(
            torch::nn::CrossEntropyLossOptions().weight(weight).ignore_index(CLASSIFICATION_IGNORE_INDEX)
          ),
          [] (torch::Tensor& logits) -> torch::Tensor {
            return logits.argmax(-1);
          }
        )
      );
      out.back().classifier.ptr()->to(torch::kCUDA);
    }
  }
  return out;
}

void runTraining(const Config& config,
                 const std::string& modelDir,
                 const std::string& dataDir,
                 std::vector<Task>& tasks,
                 int batchSize,
                 int numWorkers,
                 int numEpochs) {

	// Initialize models
  BertModel model(config);
  loadState(modelDir, *model);
	model->to(torch::kCUDA);

	//Initialize dataset
  TextDatasetType trainDataset = getDataset(modelDir, tasks, "train");
  TextDatasetType valDataset = getDataset(modelDir, tasks, "val");

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

  // Initialize criteria
  tasks = initTasks(tasks, trainDataset, config);

	// Initialize optimizer
  std::vector<torch::Tensor> parameters = model->parameters(true);
  for (auto& task : tasks) {
    std::vector<torch::Tensor> classifierParameters = task.classifier.ptr()->parameters(true);
    parameters.insert(parameters.begin(), classifierParameters.begin(), classifierParameters.end());
  }
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

    trainLoop(model, tasks, trainLoader, trainLosses, trainLabels, trainPredictions, optimizer);
    std::cout << "\tOK" << std::endl;
  
    for (size_t i = 0; i < tasks.size(); i++){
      std::cout << "Task: " << tasks[i].name << std::endl;
      float sum = 0.0f;
      for (float x : trainLosses[i] ) sum += x;
      float avg = sum / trainLosses[i].size();
      std::cout << "\tavg. loss=" << std::fixed << std::setprecision(3) << avg;
      for (const auto& metric : tasks[i].metrics) {
        float val = metric.second(trainLabels[i], trainPredictions[i]);
        std::cout << " " << metric.first << "=" << std::fixed << std::setprecision(3) << val;
      }
      std::cout << std::endl;
    }

    std::cout << "Validation..." << std::flush;
    trainLoop(model, tasks, valLoader, valLosses, valLabels, valPredictions);
    std::cout << "\tOK" << std::endl;

    for (size_t i = 0; i < tasks.size(); i++){
       std::cout << "\tTask: " << tasks[i].name << std::endl;
       float sum = 0.0f;
       for (float x : valLosses[i] ) sum += x;
       float avg = sum / valLosses[i].size();
       std::cout << "\t\tavg. loss=" << std::fixed << std::setprecision(3) << avg;
       for (const auto& metric : tasks[i].metrics) {
         float val = metric.second(valLabels[i], valPredictions[i]);
         std::cout << " " << metric.first << "=" << std::fixed << std::setprecision(3) << val;
       }
       std::cout << std::endl;
     }
	}
}
