#include "bert_intermediate.h"

BertIntermediateImpl::BertIntermediateImpl() {}
BertIntermediateImpl::BertIntermediateImpl(Config const &config)
  : dense (torch::nn::Linear(config.hiddenSize, config.intermediateSize)) {
  register_module("dense", dense);
}

torch::Tensor BertIntermediateImpl::forward(torch::Tensor hiddenStates) {
  // hiddenStates before shape: (BATCH_SIZE, MAX_SEQUENCE_LENGTH, HIDDEN_SIZE)
  hiddenStates = dense->forward(hiddenStates);
  // hiddenStates after shape:
  //   (BATCH_SIZE, MAX_SEQUENCE_LENGTH, INTERMEDIATE_SIZE)
  hiddenStates = torch::gelu(hiddenStates);
  return hiddenStates;
}



