#ifndef BERT_ENCODER_H
#define BERT_ENCODER_H
#include <torch/nn/module.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/types.h>

#include "config.h"

class BertEncoderImpl : public torch::nn::Module {
  public:
    BertEncoderImpl();
    explicit BertEncoderImpl(Config const &config);
    torch::Tensor forward(torch::Tensor hiddenStates,
                          torch::Tensor attentionMask);
	private:
		torch::nn::ModuleList layer{nullptr};
    size_t numLayers;
}; TORCH_MODULE(BertEncoder);
#endif
