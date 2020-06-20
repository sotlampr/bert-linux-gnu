#include "bert_attention.h"

BertAttentionImpl::BertAttentionImpl() {}
BertAttentionImpl::BertAttentionImpl(Config const &config)
  : self (BertSelfAttention(config)),
    output (BertSelfOutput(config)) {
  register_module("self", self);
  register_module("output", output);
}

torch::Tensor BertAttentionImpl::forward(torch::Tensor inputTensor,
                                         torch::Tensor attentionMask) {
  // inputTensor shape: (BATCH_SIZE, MAX_SEQUENCE_LENGTH, HIDDEN_SIZE)
  // attentionMask shape: (BATCH_SIZE, 1, 1, MAX_SEQUENCE_LENGTH)
  // selfOutputs shape: (BATCH_SIZE, MAX_SEQUENCE_LENGTH, HIDDEN_SIZE)
  torch::Tensor selfOutputs = self->forward(inputTensor, attentionMask);

  // attentionOutput shape: (BATCH_SIZE, MAX_SEQUENCE_LENGTH, HIDDEN_SIZE)
  torch::Tensor attentionOutput = output->forward(selfOutputs, inputTensor);
  return attentionOutput;
}
