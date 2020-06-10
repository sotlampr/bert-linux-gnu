#include "classifier.h"

BinaryClassifierImpl::BinaryClassifierImpl() {};
BinaryClassifierImpl::BinaryClassifierImpl(Config const &config, int numLabels)
  : dense (torch::nn::Linear(config.hiddenSize, numLabels)),
    dropout(torch::nn::Dropout(config.hiddenDropoutProb)) {
  register_module("dense", dense);
  register_module("dropout", dropout);
}

torch::Tensor BinaryClassifierImpl::forward(torch::Tensor hidden) {
  torch::Tensor output = dense->forward(dropout->forward(hidden));
  if (dense->options.out_features() == 1) {
    return output.squeeze(-1);
  }
  return output;
}

MulticlassClassifierImpl::MulticlassClassifierImpl() {};
MulticlassClassifierImpl::MulticlassClassifierImpl(Config const &config, int numClasses)
  : dense (torch::nn::Linear(config.hiddenSize, numClasses)),
    dropout(torch::nn::Dropout(config.hiddenDropoutProb)) {
  register_module("dense", dense);
  register_module("dropout", dropout);
}

torch::Tensor MulticlassClassifierImpl::forward(torch::Tensor hidden) {
  return dense->forward(dropout->forward(hidden));
}
