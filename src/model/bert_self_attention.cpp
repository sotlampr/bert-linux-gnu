#include "bert_self_attention.h"

#include <cmath>

BertSelfAttentionImpl::BertSelfAttentionImpl() {}

BertSelfAttentionImpl::BertSelfAttentionImpl(Config const &config)
  : numAttentionHeads (config.numAttentionHeads),
    hiddenSize (config.hiddenSize),
    attentionHeadSize (config.hiddenSize / config.numAttentionHeads),
    query (torch::nn::Linear(config.hiddenSize, config.hiddenSize)),
    key (torch::nn::Linear(config.hiddenSize, config.hiddenSize)),
    value (torch::nn::Linear(config.hiddenSize, config.hiddenSize)),
    dropout(torch::nn::Dropout(config.attentionDropoutProb)) {

  register_module("query", query);
  register_module("key", key);
  register_module("value", value);
  register_module("dropout", dropout);
}

torch::Tensor BertSelfAttentionImpl::transposeForScores(torch::Tensor x) {
  x = x.view({x.size(0), x.size(1), numAttentionHeads, attentionHeadSize});
  return x.permute({0, 2, 1, 3});
}

torch::Tensor BertSelfAttentionImpl::forward(torch::Tensor hiddenStates,
                                             torch::Tensor attentionMask) {
  // hiddenStates shape: (BATCH_SIZE, MAX_SEQUENCE_LENGTH, HIDDEN_SIZE)

  // q,k,v layers shape:
  //   (BATCH_SIZE, NUM_LAYERS, MAX_SEQUENCE_LENGTH, NUM_ATTENTION_HEADS)
  torch::Tensor queryLayer = transposeForScores(query->forward(hiddenStates));
  torch::Tensor keyLayer = transposeForScores(key->forward(hiddenStates));
  torch::Tensor valueLayer = transposeForScores(value->forward(hiddenStates));

  // Take the dot product between "query" and "key" to get the raw attention
  // scores for each layer
  // shape: (BATCH_SIZE, NUM_LAYERS, MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH)
  torch::Tensor attentionScores = torch::matmul(queryLayer, keyLayer.transpose(-1, -2));
  attentionScores /= std::sqrt(attentionHeadSize);

  attentionScores += attentionMask;

  // Convert to probabilities
  torch::Tensor attentionProbs = attentionScores.softmax(-1);
  attentionProbs = dropout->forward(attentionProbs);

  // Batch matrix multiplication for each sample and layer
  //   (MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH) ( attentionProbs)
  //    @ (MAX_SEQUENCE_LENGTH, //   NUM_ATTENTION_HEADS) (valueLayer)
  //  contextLayer shape:
  //   (BATCH_SIZE, NUM_LAYERS, MAX_SEQUENCE_LENGTH, NUM_ATTENTION_HEADS)
  torch::Tensor contextLayer = torch::matmul(attentionProbs, valueLayer);

  // Move MAX_SEQUENCE_LENGTH dimension after BATCH_SIZE:
  //   (BATCH_SIZE, MAX_SEQUENCE_LENGTH, NUM_LAYERS, NUM_ATTENTION_HEADS)
  contextLayer = contextLayer.permute({0, 2, 1, 3});

  // View as (BATCH_SIZE, MAX_SEQUENCE_LENGTH, HIDDEN_SIZE)
  contextLayer = contextLayer.reshape({contextLayer.size(0), contextLayer.size(1), hiddenSize});
  return contextLayer;
}
