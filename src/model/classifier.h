#ifndef BINARY_CLASSIFIER_H
#define BINARY_CLASSIFIER_H
#include <torch/nn/module.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/types.h>

#include "config.h"
#include "bert_pooler.h"

struct BinaryClassifierOptions {
  Config config;
  int numLabels;
  bool tokenLevel;
};

struct MutliclassClassifierOptions {
  Config config;
  int numClasses;
  bool tokenLevel;
};

class BinaryClassifierImpl : public torch::nn::Module {
  public:
    BinaryClassifierImpl();
    explicit BinaryClassifierImpl(const BinaryClassifierOptions& options);
    torch::Tensor forward(torch::Tensor hidden);
    BinaryClassifierOptions options;
  private:
    torch::nn::Linear dense{nullptr};
    torch::nn::Dropout dropout{nullptr};
    BertPooler pooler{nullptr};
}; TORCH_MODULE(BinaryClassifier);

class MulticlassClassifierImpl : public torch::nn::Module {
  public:
    MulticlassClassifierImpl();
    explicit MulticlassClassifierImpl(const MutliclassClassifierOptions& options);
    torch::Tensor forward(torch::Tensor hidden);
    MutliclassClassifierOptions options;
  private:
    torch::nn::Linear dense{nullptr};
    torch::nn::Dropout dropout{nullptr};
    BertPooler pooler{nullptr};
}; TORCH_MODULE(MulticlassClassifier);
#endif
