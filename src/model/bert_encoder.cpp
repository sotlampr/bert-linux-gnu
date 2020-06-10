#include "bert_encoder.h"

#include "bert_layer.h"

BertEncoderImpl::BertEncoderImpl() {}
BertEncoderImpl::BertEncoderImpl(Config const &config)
  : numLayers (config.numHiddenLayers), layer (torch::nn::ModuleList()) {
	for (size_t i = 0; i < config.numHiddenLayers; i++) {
		layer->push_back(BertLayer(config));
	}
  register_module("layer", layer);
}

torch::Tensor BertEncoderImpl::forward(torch::Tensor hiddenStates,
                                       torch::Tensor attentionMask) {
  // std::cout << "BertEncoder" << std::endl;
  for (const auto &module : *layer) {
    hiddenStates = module->as<BertLayer>()->forward(hiddenStates, attentionMask);
  }
  return hiddenStates;
}
