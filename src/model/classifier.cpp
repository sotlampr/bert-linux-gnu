#include "classifier.h"

BinaryClassifierImpl::BinaryClassifierImpl() {};
BinaryClassifierImpl::BinaryClassifierImpl(const BinaryClassifierOptions& options)
  : dense (torch::nn::Linear(options.config.hiddenSize, options.numLabels)),
    dropout (torch::nn::Dropout(options.config.hiddenDropoutProb)),
    pooler (BertPooler(options.config, !options.tokenLevel)),
    options (options) {
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
MulticlassClassifierImpl::MulticlassClassifierImpl(const MutliclassClassifierOptions& options)
  : dense (torch::nn::Linear(options.config.hiddenSize, options.numClasses)),
    dropout(torch::nn::Dropout(options.config.hiddenDropoutProb)),
    pooler (BertPooler(options.config, !options.tokenLevel)),
    options (options) {
  register_module("dense", dense);
  register_module("dropout", dropout);
  register_module("pooler", pooler);
}

torch::Tensor MulticlassClassifierImpl::forward(torch::Tensor hidden) {
  return dense->forward(dropout->forward(pooler->forward(hidden)));
}
