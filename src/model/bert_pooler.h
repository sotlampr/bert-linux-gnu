#ifndef BERT_POOLER_Hifndef
#define BERT_POOLER_Hifndef
#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/types.h>

#include "config.h"

class BertPoolerImpl : public torch::nn::Module {
  public:
    BertPoolerImpl();
    explicit BertPoolerImpl(Config const &config, bool useCLS);
    torch::Tensor forward(torch::Tensor hiddenStates);
  private:
    torch::nn::Linear dense{nullptr};
    bool useCLS; // Whether to use the first [CLS] token for sentence-level tasks
}; TORCH_MODULE(BertPooler);

#endif
