#ifndef TRAIN_LOOP_H
#define TRAIN_LOOP_H
#include <vector>

#include <torch/optim.h>
#include <torch/types.h>

#include "data.h"
#include "model.h"
#include "train/task.h"

// Run training for an epoch. Helper function used by `trainLoop`
inline void innerLoop(BertModel &model,
                      std::vector<Task> &tasks,
                      TextDataLoaderType &loader,
                      std::vector<std::vector<float>> &losses,
                      std::vector<torch::Tensor> &labels,
                      std::vector<torch::Tensor> &predictions,
                      std::function<void (torch::Tensor)> callback);

// Run training for an epoch.
// Writes results to the referenced losses, labels, and predictions
void trainLoop(BertModel &model,
               std::vector<Task> &tasks,
               TextDataLoaderType &loader,
               std::vector<std::vector<float>> &losses,
               std::vector<torch::Tensor> &labels,
               std::vector<torch::Tensor> &predictions,
               torch::optim::Optimizer &optimizer);

// Run vaildation for an epoch (overloaded - no optimizer argument)
void trainLoop(BertModel &model,
               std::vector<Task> &tasks,
               TextDataLoaderType &loader,
               std::vector<std::vector<float>> &losses,
               std::vector<torch::Tensor> &labels,
               std::vector<torch::Tensor> &predictions);

#endif
