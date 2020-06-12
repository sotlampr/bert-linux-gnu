#ifndef BERT_MODEL_H
#define BERT_MODEL_H
#include <torch/nn/module.h>
#include <torch/types.h>

#include "bert_embeddings.h"
#include "bert_encoder.h"
#include "config.h"

class BertModelImpl : public torch::nn::Module {
  public:
    BertModelImpl();
    explicit BertModelImpl(Config const &config);
    torch::Tensor forward(torch::Tensor inputIds);
  private:
    BertEmbeddings embeddings{nullptr};
    BertEncoder encoder{nullptr};
}; TORCH_MODULE(BertModel);
#endif
