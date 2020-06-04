#ifndef BERT_H
#define BERT_H

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
    uint32_t numAttentionHeads, hiddenSize, attentionHeadSize;
}; TORCH_MODULE(BertSelfAttention);

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

class BertIntermediateImpl : public torch::nn::Module {
  public:
    BertIntermediateImpl();
    explicit BertIntermediateImpl(Config const &config);
    torch::Tensor forward(torch::Tensor hiddenStates);
  private:
    torch::nn::Linear dense{nullptr};
}; TORCH_MODULE(BertIntermediate);

class BertOutputImpl : public torch::nn::Module {
  public:
    BertOutputImpl();
    explicit BertOutputImpl(Config const &config);
    torch::Tensor forward(torch::Tensor hiddenStates, torch::Tensor inputTensor);
  private:
    torch::nn::Linear dense{nullptr};
    torch::nn::LayerNorm layerNorm{nullptr};
    torch::nn::Dropout dropout{nullptr};
}; TORCH_MODULE(BertOutput);

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

class BertEncoderImpl : public torch::nn::Module {
  public:
    BertEncoderImpl();
    explicit BertEncoderImpl(Config const &config);
    torch::Tensor forward(torch::Tensor hiddenStates,
                          torch::Tensor attentionMask);
	private:
		torch::nn::ModuleList layer{nullptr};
    uint32_t numLayers;
}; TORCH_MODULE(BertEncoder);

class BertPoolerImpl : public torch::nn::Module {
  public:
    BertPoolerImpl();
    explicit BertPoolerImpl(Config const &config);
    torch::Tensor forward(torch::Tensor hiddenStates);
  private:
    torch::nn::Linear dense{nullptr};
}; TORCH_MODULE(BertPooler);

class BertModelImpl : public torch::nn::Module {
  public:
    BertModelImpl();
    explicit BertModelImpl(Config const &config);
    torch::Tensor forward(torch::Tensor inputIds);
  private:
    BertEmbeddings embeddings{nullptr};
    BertEncoder encoder{nullptr};
    BertPooler pooler{nullptr};
}; TORCH_MODULE(BertModel);

class BinaryClassifierImpl : public torch::nn::Module {
  public:
    BinaryClassifierImpl();
    explicit BinaryClassifierImpl(Config const &config, int numLabels = 1);
    torch::Tensor forward(torch::Tensor hidden);
  private:
    torch::nn::Linear dense{nullptr};
    torch::nn::Dropout dropout{nullptr};
}; TORCH_MODULE(BinaryClassifier);
#endif
