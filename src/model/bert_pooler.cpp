#include <torch/torch.h>

#include "config.h"
#include "bert_pooler.h"

BertPoolerImpl::BertPoolerImpl() {}
BertPoolerImpl::BertPoolerImpl(Config const &config)
  : dense (torch::nn::Linear(config.hiddenSize, config.hiddenSize)) {
  register_module("dense", dense);
}
torch::Tensor BertPoolerImpl::forward(torch::Tensor hiddenStates) {
  // std::cout << "BertPooler" << std::endl;
  // std::cout << "Pooler hiddenStates: " << hiddenStates.sizes() << std::endl;
  hiddenStates = hiddenStates.index({torch::indexing::Slice(), 0});
  // std::cout << "	after indexing: " << hiddenStates.sizes() << std::endl;
  torch::Tensor pooledOutput = dense->forward(hiddenStates);
  pooledOutput = torch::tanh(pooledOutput);
  return pooledOutput;
}
