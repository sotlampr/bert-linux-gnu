#include "classifier.h"

BinaryClassifierImpl::BinaryClassifierImpl() {};
BinaryClassifierImpl::BinaryClassifierImpl(Config const &config,
                                           int numLabels,
                                           bool tokenLevel)
  : dense (torch::nn::Linear(config.hiddenSize, numLabels)),
    dropout (torch::nn::Dropout(config.hiddenDropoutProb)),
    pooler (BertPooler(config, !tokenLevel)) {
  register_module("dense", dense);
  register_module("dropout", dropout);
  register_module("pooler", pooler);
}

torch::Tensor BinaryClassifierImpl::forward(torch::Tensor hidden) {
  torch::Tensor output = dense->forward(dropout->forward(pooler->forward(hidden)));
  if (dense->options.out_features() == 1) {
    return output.squeeze(-1);
  }
  return output;
}

MulticlassClassifierImpl::MulticlassClassifierImpl() {};
MulticlassClassifierImpl::MulticlassClassifierImpl(Config const &config,
                                                   int numClasses,
                                                   bool tokenLevel)
  : dense (torch::nn::Linear(config.hiddenSize, numClasses)),
    dropout(torch::nn::Dropout(config.hiddenDropoutProb)),
    pooler (BertPooler(config, !tokenLevel)) {
  register_module("dense", dense);
  register_module("dropout", dropout);
  register_module("pooler", pooler);
}

torch::Tensor MulticlassClassifierImpl::forward(torch::Tensor hidden) {
  return dense->forward(dropout->forward(pooler->forward(hidden)));
}
