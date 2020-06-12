#include "bert_pooler.h"

BertPoolerImpl::BertPoolerImpl() {}
BertPoolerImpl::BertPoolerImpl(Config const &config, bool useCLS)
  : dense (torch::nn::Linear(config.hiddenSize, config.hiddenSize)),
    useCLS (useCLS) {
  register_module("dense", dense);
}
torch::Tensor BertPoolerImpl::forward(torch::Tensor hiddenStates) {
  // std::cout << "BertPooler" << std::endl;
  // std::cout << "Pooler hiddenStates: " << hiddenStates.sizes() << std::endl;
  if (useCLS) hiddenStates = hiddenStates.index({torch::indexing::Slice(), 0});
  // std::cout << "	after indexing: " << hiddenStates.sizes() << std::endl;
  torch::Tensor pooledOutput = dense->forward(hiddenStates);
  pooledOutput = torch::tanh(pooledOutput);
  return pooledOutput;
}
