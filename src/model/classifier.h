#ifndef BINARY_CLASSIFIER_H
#define BINARY_CLASSIFIER_H
#include <torch/nn/module.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/types.h>

#include "config.h"
#include "bert_pooler.h"

class BinaryClassifierImpl : public torch::nn::Module {
  public:
    BinaryClassifierImpl();
    explicit BinaryClassifierImpl(Config const &config,
                                  int numLabels = 1,
                                  bool tokenLevel = false);
    torch::Tensor forward(torch::Tensor hidden);
  private:
    torch::nn::Linear dense{nullptr};
    torch::nn::Dropout dropout{nullptr};
    BertPooler pooler{nullptr};
}; TORCH_MODULE(BinaryClassifier);

class MulticlassClassifierImpl : public torch::nn::Module {
  public:
    MulticlassClassifierImpl();
    explicit MulticlassClassifierImpl(Config const &config,
                                      int numClasses,
                                      bool tokenLevel = false);
    torch::Tensor forward(torch::Tensor hidden);
  private:
    torch::nn::Linear dense{nullptr};
    torch::nn::Dropout dropout{nullptr};
    BertPooler pooler{nullptr};
}; TORCH_MODULE(MulticlassClassifier);
#endif
