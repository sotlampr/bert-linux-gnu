#include "bert_layer.h"

BertLayerImpl::BertLayerImpl() {}
BertLayerImpl::BertLayerImpl(Config const &config)
  : attention (BertAttention(config)),
    intermediate (BertIntermediate(config)),
    output (BertOutput(config)) {
  register_module("attention", attention);
  register_module("intermediate", intermediate);
  register_module("output", output);
}

torch::Tensor BertLayerImpl::forward(torch::Tensor hiddenStates,
                                     torch::Tensor attentionMask) {
  // inputIds shape: (BATCH_SIZE, MAX_SEQUENCE_LENGTH) (non-embedded ids)
  // attentionMask shape: (BATCH_SIZE, 1, 1, MAX_SEQUENCE_LENGTH)
  // attentionOutputs shape: (BATCH_SIZE, MAX_SEQUENCE_LENGTH, HIDDEN_SIZE)
  torch::Tensor attentionOutputs = attention->forward(hiddenStates, attentionMask);

  // intermediateOutput shape:
  //   (BATCH_SIZE, MAX_SEQUENCE_LENGTH, INTERMEDIATE_SIZE)
  torch::Tensor intermediateOutput = intermediate->forward(attentionOutputs);

  // layerOutput shape: (BATCH_SIZE, MAX_SEQUENCE_LENGTH, HIDDEN_SIZE)
  torch::Tensor layerOutput = output->forward(intermediateOutput, attentionOutputs);
  return layerOutput;
}
