#include <torch/torch.h>

#include "config.h"
#include "bert_intermediate.h"

BertIntermediateImpl::BertIntermediateImpl() {}
BertIntermediateImpl::BertIntermediateImpl(Config const &config)
  : dense (torch::nn::Linear(config.hiddenSize, config.intermediateSize)) {
  register_module("dense", dense);
}

torch::Tensor BertIntermediateImpl::forward(torch::Tensor hiddenStates) {
  // std::cout << "BertIntermediate" << std::endl;
  // std::cout << "	BertIntermeddiate Input size: " << hiddenStates.sizes() << std::endl;
  hiddenStates = dense->forward(hiddenStates);
  hiddenStates = torch::gelu(hiddenStates);
  // std::cout << "	BertIntermeddiate Output size: " << hiddenStates.sizes() << std::endl;
  return hiddenStates;
}



