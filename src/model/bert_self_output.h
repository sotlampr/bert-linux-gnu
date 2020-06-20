#ifndef BERT_SELF_OUTPUT_H
#define BERT_SELF_OUTPUT_H
#include <torch/nn/module.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>
#include <torch/types.h>

#include "config.h"

class BertSelfOutputImpl : public torch::nn::Module {
  public:
    BertSelfOutputImpl();
    explicit BertSelfOutputImpl(Config const &config);
    torch::Tensor forward(torch::Tensor hiddenStates,
                          torch::Tensor inputTensor);
	private:
    torch::nn::Linear dense{nullptr};
    torch::nn::LayerNorm layerNorm{nullptr};
    torch::nn::Dropout dropout{nullptr};
}; TORCH_MODULE(BertSelfOutput);

#endif
