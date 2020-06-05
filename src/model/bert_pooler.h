#ifndef BERT_POOLER_Hifndef
#define BERT_POOLER_Hifndef
#include <torch/torch.h>
#include "config.h"

class BertPoolerImpl : public torch::nn::Module {
  public:
    BertPoolerImpl();
    explicit BertPoolerImpl(Config const &config);
    torch::Tensor forward(torch::Tensor hiddenStates);
  private:
    torch::nn::Linear dense{nullptr};
}; TORCH_MODULE(BertPooler);
#endif
