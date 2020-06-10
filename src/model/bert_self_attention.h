#ifndef BERT_SELF_ATTENTION_H
#define BERT_SELF_ATTENTION_H
#include <torch/nn/module.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/types.h>

#include "config.h"
#include <torch/torch.h>
#include "config.h"

class BertSelfAttentionImpl : public torch::nn::Module {
  public:
    BertSelfAttentionImpl();
    explicit BertSelfAttentionImpl(Config const &config);
    torch::Tensor forward(torch::Tensor hiddenStates,
                          torch::Tensor attentionMask);
  private:
    torch::Tensor transposeForScores(torch::Tensor x);
    torch::nn::Linear query{nullptr}, key{nullptr}, value{nullptr};
    torch::nn::Dropout dropout{nullptr};
    int numAttentionHeads, hiddenSize, attentionHeadSize;
}; TORCH_MODULE(BertSelfAttention);
#endif
