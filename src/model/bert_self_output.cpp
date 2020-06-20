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
  // hiddenStates shape: (BATCH_SIZE, MAX_SEQUENCE_LENGTH, HIDDEN_SIZE)
  hiddenStates = dense->forward(hiddenStates);
  hiddenStates = dropout->forward(hiddenStates);

  // Add input (residual)
  hiddenStates = layerNorm->forward(hiddenStates + inputTensor);

  return hiddenStates;
}
