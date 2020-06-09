#ifndef TRAIN_LOOP_H
#define TRAIN_LOOP_H
#include <tuple>
#include <vector>
#include <torch/torch.h>
#include "model.h"
#include "train/task.h"

inline void innerLoop(BertModel &model,
                      std::vector<Task> &tasks,
                      TextDataLoaderType &loader,
                      std::vector<std::vector<float>> &losses,
                      std::vector<torch::Tensor> &labels,
                      std::vector<torch::Tensor> &predictions,
                      std::function<void (torch::Tensor)> callback);

void trainLoop(BertModel &model,
               std::vector<Task> &tasks,
               TextDataLoaderType &loader,
               std::vector<std::vector<float>> &losses,
               std::vector<torch::Tensor> &labels,
               std::vector<torch::Tensor> &predictions,
               torch::optim::Optimizer &optimizer);

void trainLoop(BertModel &model,
               std::vector<Task> &tasks,
               TextDataLoaderType &loader,
               std::vector<std::vector<float>> &losses,
               std::vector<torch::Tensor> &labels,
               std::vector<torch::Tensor> &predictions);

#endif
