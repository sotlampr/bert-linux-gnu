#ifndef TRAIN_LOOP_H
#define TRAIN_LOOP_H
#include <tuple>
#include <vector>
#include <torch/torch.h>
#include "model.h"

inline void innerLoop(TextDataLoaderType &loader,
                      BertModel &model,
                      BinaryClassifier &classifier,
                      torch::nn::BCEWithLogitsLoss &criterion,
                      std::vector<float> &losses,
                      torch::Tensor &labels,
                      torch::Tensor &predictions,
                      std::function<void (torch::Tensor)> callback);

void trainLoop(BertModel &model,
               BinaryClassifier &classifier,
               TextDataLoaderType &loader,
               torch::nn::BCEWithLogitsLoss &criterion,
               torch::optim::Optimizer &optimizer,
               std::vector<float> &losses,
               torch::Tensor &labels,
               torch::Tensor &predictions);

void trainLoop(BertModel &model,
               BinaryClassifier &classifier,
               TextDataLoaderType &loader,
               torch::nn::BCEWithLogitsLoss &criterion,
               std::vector<float> &losses,
               torch::Tensor &labels,
               torch::Tensor &predictions);

#endif
