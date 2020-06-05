#include <torch/torch.h>

#include "config.h"
#include "bert_self_output.h"

BertSelfOutputImpl::BertSelfOutputImpl() {}

BertSelfOutputImpl::BertSelfOutputImpl(Config const &config)
  : dense (torch::nn::Linear(config.hiddenSize, config.hiddenSize)),
    layerNorm (torch::nn::LayerNormOptions({config.hiddenSize}).eps(LAYER_NORM_EPS)),
    dropout (torch::nn::Dropout(config.hiddenDropoutProb)) {
  register_module("dense", dense);
  register_module("layerNorm", layerNorm);
  register_module("dropout", dropout);
}

torch::Tensor BertSelfOutputImpl::forward(torch::Tensor hiddenStates,
                                          torch::Tensor inputTensor) {
  // std::cout << "BertSelfOutput" << std::endl;
  // std::cout << "	BertSelfOutput Input size: " << hiddenStates.sizes() << std::endl;
  hiddenStates = dense->forward(hiddenStates);
  hiddenStates = dropout->forward(hiddenStates);
  hiddenStates = layerNorm->forward(hiddenStates + inputTensor);
  // std::cout << "	BertSelfOutput Output size: " << hiddenStates.sizes() << std::endl;
  return hiddenStates;
}
