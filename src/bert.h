#ifndef BERT_H
#define BERT_H

#include <torch/torch.h>
#include "config.h"

class BertEmbeddingsImpl : public torch::nn::Module {
  public:
    BertEmbeddingsImpl();
    BertEmbeddingsImpl(Config config);
    torch::Tensor forward(torch::Tensor inputIds,
                          torch::Tensor tokenTypeIds,
                          torch::Tensor positionIds);
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
    BertSelfAttentionImpl(Config config);
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
    BertSelfOutputImpl(Config config);
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
    BertAttentionImpl(Config config);
    torch::Tensor forward(torch::Tensor inputTensor,
                          torch::Tensor attentionMask);
	private:
		 BertSelfAttention self{nullptr};
		 BertSelfOutput output{nullptr};
}; TORCH_MODULE(BertAttention);

class BertIntermediateImpl : public torch::nn::Module {
  public:
    BertIntermediateImpl();
    BertIntermediateImpl(Config config);
    torch::Tensor forward(torch::Tensor hiddenStates);
  private:
    torch::nn::Linear dense{nullptr};
}; TORCH_MODULE(BertIntermediate);

class BertOutputImpl : public torch::nn::Module {
  public:
    BertOutputImpl();
    BertOutputImpl(Config config);
    torch::Tensor forward(torch::Tensor hiddenStates, torch::Tensor inputTensor);
  private:
    torch::nn::Linear dense{nullptr};
    torch::nn::LayerNorm layerNorm{nullptr};
    torch::nn::Dropout dropout{nullptr};
}; TORCH_MODULE(BertOutput);

class BertLayerImpl : public torch::nn::Module {
  public:
    BertLayerImpl();
    BertLayerImpl(Config config);
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
    BertEncoderImpl(Config config);
    torch::Tensor forward(torch::Tensor hiddenStates,
                          torch::Tensor attentionMask);
	private:
		torch::nn::ModuleList layer{nullptr};
    uint32_t numLayers;
}; TORCH_MODULE(BertEncoder);

class BertPoolerImpl : public torch::nn::Module {
  public:
    BertPoolerImpl();
    BertPoolerImpl(Config config);
    torch::Tensor forward(torch::Tensor hiddenStates);
  private:
    torch::nn::Linear dense{nullptr};
}; TORCH_MODULE(BertPooler);

class BertModelImpl : public torch::nn::Module {
  public:
    BertModelImpl();
    BertModelImpl(Config config);
    torch::Tensor forward(torch::Tensor inputIds,
                          torch::Tensor tokenTypeIds,
                          torch::Tensor attentionMask,
                          torch::Tensor positionIds);
  private:
    BertEmbeddings embeddings{nullptr};
    BertEncoder encoder{nullptr};
    BertPooler pooler{nullptr};
}; TORCH_MODULE(BertModel);
#endif
