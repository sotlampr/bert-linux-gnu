#include <cmath>
#include "torch/torch.h"
#include "config.h"
#include "bert.h"

BertEmbeddingsImpl::BertEmbeddingsImpl() {};

BertEmbeddingsImpl::BertEmbeddingsImpl(Config config)
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

torch::Tensor BertEmbeddingsImpl::forward(torch::Tensor inputIds,
                                          torch::Tensor tokenTypeIds,
                                          torch::Tensor positionIds) {
  torch::Tensor wordEmbed = wordEmbeddings->forward(inputIds);
  torch::Tensor posEmbed = positionEmbeddings->forward(positionIds);
  torch::Tensor tokEmbed = tokenTypeEmbeddings->forward(tokenTypeIds);
  torch::Tensor output = wordEmbed + posEmbed + tokEmbed;
  output = layerNorm->forward(output);
  output = dropout->forward(output);
  return output;
}

BertSelfAttentionImpl::BertSelfAttentionImpl() {}

BertSelfAttentionImpl::BertSelfAttentionImpl(Config config)
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
  torch::Tensor queryLayer = transposeForScores(query->forward(hiddenStates));
  torch::Tensor keyLayer = transposeForScores(key->forward(hiddenStates));
  torch::Tensor valueLayer = transposeForScores(value->forward(hiddenStates));

  torch::Tensor attentionScores = torch::matmul(queryLayer, keyLayer.transpose(-1, -2));
  attentionScores /= std::sqrt(attentionHeadSize);
  attentionScores += attentionMask;
  torch::Tensor attentionProbs = attentionScores.softmax(-1);
  attentionProbs = dropout->forward(attentionProbs);

  torch::Tensor contextLayer = torch::matmul(attentionProbs, valueLayer);
  contextLayer = contextLayer.permute({0, 2, 1, 3});
  contextLayer = contextLayer.view({contextLayer.size(0), hiddenSize});
  return contextLayer;
}

BertSelfOutputImpl::BertSelfOutputImpl() {}

BertSelfOutputImpl::BertSelfOutputImpl(Config config)
  : dense (torch::nn::Linear(config.hiddenSize, config.hiddenSize)),
    layerNorm (torch::nn::LayerNormOptions({config.hiddenSize}).eps(LAYER_NORM_EPS)),
    dropout (torch::nn::Dropout(config.hiddenDropoutProb)) {
  register_module("dense", dense);
  register_module("layerNorm", layerNorm);
  register_module("dropout", dropout);
}

torch::Tensor BertSelfOutputImpl::forward(torch::Tensor hiddenStates,
                                          torch::Tensor inputTensor) {
  hiddenStates = dense->forward(hiddenStates);
  hiddenStates = dropout->forward(hiddenStates);
  hiddenStates = layerNorm->forward(hiddenStates + inputTensor);
  return hiddenStates;
}

BertAttentionImpl::BertAttentionImpl() {}
BertAttentionImpl::BertAttentionImpl(Config config)
  : self (BertSelfAttention(config)),
    output (BertSelfOutput(config)) {
  register_module("self", self);
  register_module("output", output);
}

torch::Tensor BertAttentionImpl::forward(torch::Tensor inputTensor,
                                         torch::Tensor attentionMask) {
  torch::Tensor selfOutputs = self->forward(inputTensor, attentionMask);
  torch::Tensor attentionOutput = output->forward(selfOutputs[0], inputTensor);
  return attentionOutput;
}

BertIntermediateImpl::BertIntermediateImpl() {}
BertIntermediateImpl::BertIntermediateImpl(Config config)
  : dense (torch::nn::Linear(config.hiddenSize, config.intermediateSize)) {
  register_module("dense", dense);
}

torch::Tensor BertIntermediateImpl::forward(torch::Tensor hiddenStates) {
  hiddenStates = dense->forward(hiddenStates);
  // TODO: gelu
  hiddenStates = torch::relu(hiddenStates);
  return hiddenStates;
}


BertOutputImpl::BertOutputImpl() {}
BertOutputImpl::BertOutputImpl(Config config)
  : dense (torch::nn::Linear(config.intermediateSize, config.hiddenSize)),
    layerNorm (torch::nn::LayerNormOptions({config.hiddenSize}).eps(LAYER_NORM_EPS)),
    dropout (torch::nn::Dropout(config.hiddenDropoutProb)) {
  register_module("dense", dense);
  register_module("layerNorm", layerNorm);
  register_module("dropout", dropout);
}

torch::Tensor BertOutputImpl::forward(torch::Tensor hiddenStates,
                                      torch::Tensor inputTensor) {
  hiddenStates = dense->forward(hiddenStates);
  hiddenStates = dropout->forward(hiddenStates);
  hiddenStates = layerNorm->forward(hiddenStates + inputTensor);
  return hiddenStates;
}

BertLayerImpl::BertLayerImpl() {}
BertLayerImpl::BertLayerImpl(Config config)
  : attention (BertAttention(config)),
    intermediate (BertIntermediate(config)),
    output (BertOutput(config)) {
  register_module("attention", attention);
  register_module("intermediate", intermediate);
  register_module("output", output);
}

torch::Tensor BertLayerImpl::forward(torch::Tensor hiddenStates,
                                     torch::Tensor attentionMask) {
  torch::Tensor attentionOutputs = attention->forward(hiddenStates, attentionMask);
  torch::Tensor attentionOutput = attentionOutputs[0];
  torch::Tensor intermediateOutput = intermediate->forward(attentionOutput);
  torch::Tensor layerOutput = output->forward(intermediateOutput, attentionOutput);
  return layerOutput;
}

BertEncoderImpl::BertEncoderImpl() {}
BertEncoderImpl::BertEncoderImpl(Config config)
  : numLayers (config.numHiddenLayers) {
  torch::nn::ModuleList layer;
	for (uint32_t i = 0; i < config.numHiddenLayers; i++) {
		layer->push_back(BertLayer(config));
	}
  register_module("layer", layer);
}

torch::Tensor BertEncoderImpl::forward(torch::Tensor hiddenStates,
                                       torch::Tensor attentionMask) {
  for (const auto &module : *layer) {
    hiddenStates = module->as<BertLayer>()->forward(hiddenStates, attentionMask);
  }
  return hiddenStates;
}

BertPoolerImpl::BertPoolerImpl() {}
BertPoolerImpl::BertPoolerImpl(Config config)
  : dense (torch::nn::Linear(config.hiddenSize, config.hiddenSize)) {
  register_module("dense", dense);
}
torch::Tensor BertPoolerImpl::forward(torch::Tensor hiddenStates) {
  hiddenStates = hiddenStates.index({"...", torch::indexing::Slice(0)});
  torch::Tensor pooledOutput = dense->forward(hiddenStates);
  pooledOutput = torch::tanh(pooledOutput);
  return pooledOutput;
}

BertModelImpl::BertModelImpl() {}
BertModelImpl::BertModelImpl(Config config)
  : embeddings (BertEmbeddings(config)),
    encoder (BertEncoder(config)),
    pooler (BertPooler(config)) {
  register_module("embeddings", embeddings);
  register_module("encoder", encoder);
  register_module("pooler", pooler);
}

torch::Tensor BertModelImpl::forward(torch::Tensor inputIds,
                                     torch::Tensor tokenTypeIds,
                                     torch::Tensor attentionMask,
                                     torch::Tensor positionIds) {
  torch::Tensor extendedAttentionMask = attentionMask.unsqueeze(1).unsqueeze(2);
  extendedAttentionMask = (1.0f - extendedAttentionMask) * -10000.0f;
  torch::Tensor embeddingOutput = embeddings(inputIds, positionIds, tokenTypeIds);
  torch::Tensor encoderOutputs = encoder(embeddingOutput, extendedAttentionMask);
  torch::Tensor sequenceOutput = encoderOutputs[0];
  torch::Tensor pooledOutput = pooler(sequenceOutput);
  return pooledOutput;
}
