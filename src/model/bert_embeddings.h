#ifndef BERT_EMBEDDINGS_H
#define BERT_EMBEDDINGS_H
#include <torch/torch.h>
#include "config.h"

class BertEmbeddingsImpl : public torch::nn::Module {
  public:
    BertEmbeddingsImpl();
    explicit BertEmbeddingsImpl(Config const &config);
    torch::Tensor forward(torch::Tensor inputIds);
  private:
    torch::nn::Embedding wordEmbeddings{nullptr},
                         positionEmbeddings{nullptr},
                         tokenTypeEmbeddings{nullptr};
    torch::nn::LayerNorm layerNorm{nullptr};
    torch::nn::Dropout dropout{nullptr};
}; TORCH_MODULE(BertEmbeddings);
#endif
