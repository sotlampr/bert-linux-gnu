#include "bert_pooler.h"

BertPoolerImpl::BertPoolerImpl() {}
BertPoolerImpl::BertPoolerImpl(Config const &config, bool useCLS)
  : dense (torch::nn::Linear(config.hiddenSize, config.hiddenSize)),
    useCLS (useCLS) {
  register_module("dense", dense);
}
torch::Tensor BertPoolerImpl::forward(torch::Tensor hiddenStates) {
  // hiddenStates shape: (BATCH_SIZE, MAX_SEQUENCE_LENGTH, HIDDEN_SIZE)
  // Get the first column ([CLS] token) if useCLS
  if (useCLS) hiddenStates = hiddenStates.index({torch::indexing::Slice(), 0});
  // output shape: (BATCH_SIZE, HIDDEN_SIZE) if useCLS
  //   else (BATCH_SIZE, MAX_SEQUENCE_LENGTH, HIDDEN_SIZE)
  torch::Tensor pooledOutput = dense->forward(hiddenStates);
  pooledOutput = torch::tanh(pooledOutput);

  return pooledOutput;
}
