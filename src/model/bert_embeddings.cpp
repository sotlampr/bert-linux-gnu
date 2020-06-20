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
  // inputIds shape: (BATCH_SIZE, MAX_SEQUENCE_LENGTH)
  // TODO: detect presence of [SEP] and modify tokenTypeIds appropriately
  torch::Tensor tokenTypeIds = torch::zeros_like(inputIds).cuda();

  torch::Tensor positionIds = torch::arange(
    MAX_SEQUENCE_LENGTH,
    torch::TensorOptions().dtype(torch::kInt64)
  ).cuda().unsqueeze(0).expand_as(inputIds);

  torch::Tensor wordEmbed = wordEmbeddings->forward(inputIds);
  torch::Tensor posEmbed = positionEmbeddings->forward(positionIds);
  torch::Tensor tokEmbed = tokenTypeEmbeddings->forward(tokenTypeIds);
  // output / *Embed shape: (BATCH_SIZE, MAX_SEQUENCE_LENGTH, HIDDEN_SIZE)
  torch::Tensor output = wordEmbed + posEmbed + tokEmbed;

  output = layerNorm->forward(output);
  output = dropout->forward(output);

  return output;
}
