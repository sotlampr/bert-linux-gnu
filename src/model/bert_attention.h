#ifndef BERT_ATTENTION_H
#define BERT_ATTENTION_H
#include <torch/torch.h>
#include "config.h"
#include "bert_self_attention.h"
#include "bert_self_output.h"

class BertAttentionImpl : public torch::nn::Module {
  public:
    BertAttentionImpl();
    explicit BertAttentionImpl(Config const &config);
    torch::Tensor forward(torch::Tensor inputTensor,
                          torch::Tensor attentionMask);
	private:
		 BertSelfAttention self{nullptr};
		 BertSelfOutput output{nullptr};
}; TORCH_MODULE(BertAttention);
#endif
