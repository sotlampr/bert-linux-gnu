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
  // std::cout << "BertLayer" << std::endl;
  torch::Tensor attentionOutputs = attention->forward(hiddenStates, attentionMask);
  torch::Tensor intermediateOutput = intermediate->forward(attentionOutputs);
  torch::Tensor layerOutput = output->forward(intermediateOutput, attentionOutputs);
  // std::cout << "	BertLayer Output size: " << layerOutput.sizes() << std::endl;
  return layerOutput;
}
