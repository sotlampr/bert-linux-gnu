#include <torch/torch.h>

#include "config.h"
#include "bert_attention.h"
#include "bert_self_attention.h"
#include "bert_self_output.h"

BertAttentionImpl::BertAttentionImpl() {}
BertAttentionImpl::BertAttentionImpl(Config const &config)
  : self (BertSelfAttention(config)),
    output (BertSelfOutput(config)) {
  register_module("self", self);
  register_module("output", output);
}

torch::Tensor BertAttentionImpl::forward(torch::Tensor inputTensor,
                                         torch::Tensor attentionMask) {
  // std::cout << "BertAttention" << std::endl;
  torch::Tensor selfOutputs = self->forward(inputTensor, attentionMask);
  torch::Tensor attentionOutput = output->forward(selfOutputs, inputTensor);
  // std::cout << "	BertAttention Output size: " << attentionOutput.sizes() << std::endl;
  return attentionOutput;
}
