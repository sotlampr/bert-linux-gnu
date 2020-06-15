#include "train_utils.h"

#include <string>
#include <tuple>
#include <vector>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <fstream>

#include "model.h"
#include "optim.h"
#include "state.h"
#include "metrics.h"
#include "train_loop.h"


std::vector<Task> initTasks(std::vector<Task>& tasks,
                            TextDatasetType& dataset,
                            const Config& config,
                            const std::string& saveFname) {
  std::vector<Task> out;
  std::vector<torch::Tensor> weights = dataset.dataset().getClassWeights(tasks);

  for (size_t i = 0; i< tasks.size(); i++) {
    bool tokenLevel = (TokenLevel & tasks[i].taskType) == TokenLevel;
    if ((Regression & tasks[i].taskType) == Regression) {
      throw std::runtime_error("Regression not implemented");
    } else if ((Binary & tasks[i].taskType) == Binary) {
      torch::Tensor pos_weight = weights[i];
      BinaryClassifierOptions options{config, static_cast<int>(pos_weight.size(0)), tokenLevel};
      if (!saveFname.empty()) saveStruct(options, saveFname + "-" + tasks[i].name + "-binary.config");
      out.push_back(
        Task(
          tasks[i], BinaryClassifier(options),
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
      MutliclassClassifierOptions options{config, static_cast<int>(weight.size(0)), tokenLevel};
      if (!saveFname.empty()) saveStruct(options, saveFname + "-" + tasks[i].name + "-multiclass.config");
      out.push_back(
        Task(
          tasks[i], MulticlassClassifier(options),
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

void saveModel(BertModel &model,
               std::vector<Task> &tasks,
               const std::string& baseFname,
               float& currentMetric,
               float& bestMetric) {
  if (currentMetric > bestMetric) {
    std::cerr << "Model improved, saving..." << std::endl;
    bestMetric = currentMetric;
    torch::save(model, baseFname + "-bert.pt");
    for (const auto& task : tasks) {
      std::string moduleName = task.classifier.ptr()->name();
      std::string moduleId;
      if (moduleName == "MulticlassClassifierImpl") {
        moduleId = "multiclass";
      } else if (moduleName == "BinaryClassifierImpl") {
        moduleId = "binary";
      }
      torch::save(task.classifier.ptr(), baseFname + "-" + task.name + "-" + moduleId + ".pt");
    }
  }
}


void runTraining(const std::string& modelDir,
                 const std::string& dataDir,
                 std::vector<Task>& tasks,
                 int batchSize,
                 int numWorkers,
                 int numEpochs,
                 const std::string& saveFname,
                 int randomSeed) {
  torch::manual_seed(randomSeed);

  // Read config
  Config config;
  readStruct(config, modelDir + "/config");

  // Save Config if saveFname
  if (!saveFname.empty()) {
    saveStruct(config, saveFname + "-bert.config");
  }

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
  tasks = initTasks(tasks, trainDataset, config, saveFname);

  // Initialize optimizer
  std::vector<torch::Tensor> dParams;
  std::vector<torch::Tensor> ndParams;
  std::cout << "Gathering bertModel's parameters..." << std::endl;
  for (const auto& param : model->named_parameters()) {
    const auto& name = param.key();
    if ((name.find("bias") != std::string::npos)
        || (name.find("layerNorm.weight") != std::string::npos)) {
      ndParams.push_back(param.value());
    } else {
      dParams.push_back(param.value());
    }
  }

  for (auto& task : tasks) {
    for (const auto& param : task.classifier.ptr()->named_parameters()) {
      const auto& name = param.key();
      if ((name.find("bias") != std::string::npos)
          || (name.find("layerNorm.weight") != std::string::npos)) {
        ndParams.push_back(param.value());
      } else {
        dParams.push_back(param.value());
      }
    };
  }

  std::vector<torch::optim::OptimizerParamGroup> param_groups {
    torch::optim::OptimizerParamGroup(
      ndParams,
      std::make_unique<torch::optim::AdamWOptions>(
        torch::optim::AdamWOptions(1e-5)
      )
    ),
    torch::optim::OptimizerParamGroup(
      dParams,
      std::make_unique<torch::optim::AdamWOptions>(
        torch::optim::AdamWOptions(1e-5).weight_decay(WEIGHT_DECAY)
      )
    )
  };

  torch::optim::AdamW optimizer(param_groups);

  float bestMetric = -std::numeric_limits<float>::infinity();
  float currentMetric;

  // Print headers
  std::cout << "epoch";
  for (auto const& task : tasks){
    std::cout << "," << task.name << "_train_loss";
    for (const auto& metric : task.metrics) {
      std::cout << "," << task.name << "_train_" << metric.first;
    }
  }

  for (auto const& task : tasks){
    std::cout << "," <<  task.name << "_val_loss";
    for (const auto& metric : task.metrics) {
      std::cout << "," <<  task.name << "_val_" << metric.first;
    }
  }
  std::cout << std::endl;

  for (int epoch=1; epoch <= numEpochs; epoch++) {
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
  
    std::cout << epoch;
    for (size_t i = 0; i < tasks.size(); i++){
      float sum = 0.0f;
      for (float x : trainLosses[i] ) sum += x;
      float avg = sum / trainLosses[i].size();
      std::cout << "," << avg;
      for (const auto& metric : tasks[i].metrics) {
        // std::cout << std::endl << "task: " << tasks[i].name << ", metric: " << metric.first << std::endl;
        float val = metric.second(trainLabels[i], trainPredictions[i]);
        std::cout << "," << val;
        // std::cout << std::endl;
      }
    }

    trainLoop(model, tasks, valLoader, valLosses, valLabels, valPredictions);

    for (size_t i = 0; i < tasks.size(); i++){
      float sum = 0.0f;
      for (float x : valLosses[i] ) sum += x;
      float avg = sum / valLosses[i].size();
      std::cout << "," << avg;
      for (const auto& metric : tasks[i].metrics) {
        // std::cout << std::endl << "task: " << tasks[i].name << ", metric: " << metric.first << std::endl;
        float val = metric.second(valLabels[i], valPredictions[i]);
        std::cout << "," << val;
        // std::cout << std::endl;
      }
    }
    std::cout << std::endl;
    // Save model if applicable
    if (!saveFname.empty()) {
      currentMetric = tasks[0].metrics[0].second(valLabels[0], valPredictions[0]);
      saveModel(model, tasks, saveFname, currentMetric, bestMetric);
    }
  }
}
