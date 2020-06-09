#include <torch/torch.h>

#include "config.h"
#include "classifier.h"

BinaryClassifierImpl::BinaryClassifierImpl() {};
BinaryClassifierImpl::BinaryClassifierImpl(Config const &config, int numLabels)
  : dense (torch::nn::Linear(config.hiddenSize, numLabels)),
    dropout(torch::nn::Dropout(config.hiddenDropoutProb)) {
  register_module("dense", dense);
  register_module("dropout", dropout);
}

torch::Tensor BinaryClassifierImpl::forward(torch::Tensor hidden) {
  return dense->forward(dropout->forward(hidden));
}
