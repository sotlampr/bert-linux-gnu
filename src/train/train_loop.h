#ifndef TRAIN_LOOP_H
#define TRAIN_LOOP_H
#include <tuple>
#include <vector>
#include <torch/torch.h>
#include "model.h"

torch::Tensor doStep(torch::data::Example<> &batch,
                     BertModel &model,
                     BinaryClassifier &classifier,
                     torch::nn::BCEWithLogitsLoss &criterion,
                     std::vector<float> &losses);

std::tuple<std::vector<float>, std::vector<torch::Tensor>>
  trainLoop(BertModel &model,
            BinaryClassifier &classifier,
            TextDataLoaderType &loader,
            torch::nn::BCEWithLogitsLoss &criterion,
            torch::optim::Optimizer &optimizer);

std::tuple<std::vector<float>, std::vector<torch::Tensor>>
  trainLoop(BertModel &model,
            BinaryClassifier &classifier,
            TextDataLoaderType &loader,
            torch::nn::BCEWithLogitsLoss &criterion);

#endif
