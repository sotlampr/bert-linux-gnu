#ifndef BERT_INTERMEDIATE_H
#define BERT_INTERMEDIATE_H
#include <torch/torch.h>
#include "config.h"

class BertIntermediateImpl : public torch::nn::Module {
  public:
    BertIntermediateImpl();
    explicit BertIntermediateImpl(Config const &config);
    torch::Tensor forward(torch::Tensor hiddenStates);
  private:
    torch::nn::Linear dense{nullptr};
}; TORCH_MODULE(BertIntermediate);
#endif
