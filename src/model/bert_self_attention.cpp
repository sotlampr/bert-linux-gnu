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
  // std::cout << "BertSelfAttention" << std::endl;
  // std::cout << "Transposing queryLayer" << std::endl;
  torch::Tensor queryLayer = transposeForScores(query->forward(hiddenStates));
  // std::cout << "Transposing keyLayer" << std::endl;
  torch::Tensor keyLayer = transposeForScores(key->forward(hiddenStates));
  // std::cout << "Transposing valueLayer" << std::endl;
  torch::Tensor valueLayer = transposeForScores(value->forward(hiddenStates));

  // std::cout << "Matmul query*key" << std::endl;
  torch::Tensor attentionScores = torch::matmul(queryLayer, keyLayer.transpose(-1, -2));
  // std::cout << "Divide by sqrt attentionHeadSize" << std::endl;
  attentionScores /= std::sqrt(attentionHeadSize);
  // std::cout << "Adding mask" << std::endl;
  // std::cout << attentionScores.sizes() << " " << attentionMask.sizes() << std::endl;
  attentionScores += attentionMask;
  // std::cout << "Applying softmax" << std::endl;
  torch::Tensor attentionProbs = attentionScores.softmax(-1);
  attentionProbs = dropout->forward(attentionProbs);

  // std::cout << "matmul attention w/ values" << std::endl;
  torch::Tensor contextLayer = torch::matmul(attentionProbs, valueLayer);
  // std::cout << "Permuting context layer" << std::endl;
  contextLayer = contextLayer.permute({0, 2, 1, 3});
  // std::cout << "Viewing context layer" << std::endl;
  contextLayer = contextLayer.reshape({contextLayer.size(0), contextLayer.size(1), hiddenSize});
  return contextLayer;
}
