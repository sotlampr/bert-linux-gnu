#include "bert_output.h"

BertOutputImpl::BertOutputImpl() {}
BertOutputImpl::BertOutputImpl(Config const &config)
  : dense (torch::nn::Linear(config.intermediateSize, config.hiddenSize)),
    layerNorm (torch::nn::LayerNormOptions({config.hiddenSize}).eps(LAYER_NORM_EPS)),
    dropout (torch::nn::Dropout(config.hiddenDropoutProb)) {
  register_module("dense", dense);
  register_module("layerNorm", layerNorm);
  register_module("dropout", dropout);
}

torch::Tensor BertOutputImpl::forward(torch::Tensor hiddenStates,
                                      torch::Tensor inputTensor) {
  // hiddenStates shape: (BATCH_SIZE, MAX_SEQUENCE_LENGTH, HIDDEN_SIZE)
  hiddenStates = dense->forward(hiddenStates);
  hiddenStates = dropout->forward(hiddenStates);
  hiddenStates = layerNorm->forward(hiddenStates + inputTensor);

  return hiddenStates;
}
