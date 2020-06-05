#include <torch/torch.h>

#include "config.h"
#include "bert_embeddings.h"

BertEmbeddingsImpl::BertEmbeddingsImpl() {};

BertEmbeddingsImpl::BertEmbeddingsImpl(Config const &config)
  : wordEmbeddings(torch::nn::EmbeddingOptions(config.vocabSize, config.hiddenSize).padding_idx(PADDING_IDX)),
    positionEmbeddings(torch::nn::EmbeddingOptions(config.maxPositionEmbeddings, config.hiddenSize)),
    tokenTypeEmbeddings(torch::nn::EmbeddingOptions(config.typeVocabSize, config.hiddenSize)),
    layerNorm(torch::nn::LayerNormOptions({config.hiddenSize}).eps(LAYER_NORM_EPS)),
    dropout(torch::nn::Dropout(config.hiddenDropoutProb)) {
  register_module("wordEmbeddings", wordEmbeddings);
  register_module("positionEmbeddings", positionEmbeddings);
  register_module("tokenTypeEmbeddings", tokenTypeEmbeddings);
  register_module("layerNorm", layerNorm);
  register_module("dropout", dropout);
}

torch::Tensor BertEmbeddingsImpl::forward(torch::Tensor inputIds) {
  // std::cout << "BertEmbeddings" << std::endl;
  // std::cout << "tokenTypeIds" << std::endl;
  torch::Tensor tokenTypeIds = torch::zeros_like(inputIds).to(torch::kCUDA);
  // std::cout << "positionIds init" << std::endl;
  torch::Tensor positionIds = torch::arange(MAX_SEQUENCE_LENGTH,
                                            torch::TensorOptions().dtype(torch::kInt64)).to(torch::kCUDA);
  // std::cout << "positionIds unsqueezing" << std::endl;
  // std::cout << "Position IDS: " << positionIds.sizes() << std::endl;
  // std::cout << "Inpuy IDS: " << inputIds.sizes() << std::endl;
  positionIds = positionIds.unsqueeze(0).expand_as(inputIds);


  torch::Tensor wordEmbed = wordEmbeddings->forward(inputIds);
  torch::Tensor posEmbed = positionEmbeddings->forward(positionIds);
  torch::Tensor tokEmbed = tokenTypeEmbeddings->forward(tokenTypeIds);
  torch::Tensor output = wordEmbed + posEmbed + tokEmbed;
  output = layerNorm->forward(output);
  output = dropout->forward(output);
  return output;
}
