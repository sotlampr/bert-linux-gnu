#ifndef BERT_LAYER_H
#define BERT_LAYER_H
#include <torch/nn/module.h>
#include <torch/types.h>

#include "bert_attention.h"
#include "bert_intermediate.h"
#include "bert_output.h"
#include "config.h"

class BertLayerImpl : public torch::nn::Module {
  public:
    BertLayerImpl();
    explicit BertLayerImpl(Config const &config);
    torch::Tensor forward(torch::Tensor hiddenStates,
                          torch::Tensor attentionMask);
  private:
    BertAttention attention{nullptr};
    BertIntermediate intermediate{nullptr};
    BertOutput output{nullptr};
}; TORCH_MODULE(BertLayer);
#endif
