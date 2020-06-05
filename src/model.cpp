#include <cmath>
#include "torch/torch.h"
#include "config.h"
#include "model.h"

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

BertSelfOutputImpl::BertSelfOutputImpl() {}

BertSelfOutputImpl::BertSelfOutputImpl(Config const &config)
  : dense (torch::nn::Linear(config.hiddenSize, config.hiddenSize)),
    layerNorm (torch::nn::LayerNormOptions({config.hiddenSize}).eps(LAYER_NORM_EPS)),
    dropout (torch::nn::Dropout(config.hiddenDropoutProb)) {
  register_module("dense", dense);
  register_module("layerNorm", layerNorm);
  register_module("dropout", dropout);
}

torch::Tensor BertSelfOutputImpl::forward(torch::Tensor hiddenStates,
                                          torch::Tensor inputTensor) {
  // std::cout << "BertSelfOutput" << std::endl;
  // std::cout << "	BertSelfOutput Input size: " << hiddenStates.sizes() << std::endl;
  hiddenStates = dense->forward(hiddenStates);
  hiddenStates = dropout->forward(hiddenStates);
  hiddenStates = layerNorm->forward(hiddenStates + inputTensor);
  // std::cout << "	BertSelfOutput Output size: " << hiddenStates.sizes() << std::endl;
  return hiddenStates;
}

BertAttentionImpl::BertAttentionImpl() {}
BertAttentionImpl::BertAttentionImpl(Config const &config)
  : self (BertSelfAttention(config)),
    output (BertSelfOutput(config)) {
  register_module("self", self);
  register_module("output", output);
}

torch::Tensor BertAttentionImpl::forward(torch::Tensor inputTensor,
                                         torch::Tensor attentionMask) {
  // std::cout << "BertAttention" << std::endl;
  torch::Tensor selfOutputs = self->forward(inputTensor, attentionMask);
  torch::Tensor attentionOutput = output->forward(selfOutputs, inputTensor);
  // std::cout << "	BertAttention Output size: " << attentionOutput.sizes() << std::endl;
  return attentionOutput;
}

BertIntermediateImpl::BertIntermediateImpl() {}
BertIntermediateImpl::BertIntermediateImpl(Config const &config)
  : dense (torch::nn::Linear(config.hiddenSize, config.intermediateSize)) {
  register_module("dense", dense);
}

torch::Tensor BertIntermediateImpl::forward(torch::Tensor hiddenStates) {
  // std::cout << "BertIntermediate" << std::endl;
  // std::cout << "	BertIntermeddiate Input size: " << hiddenStates.sizes() << std::endl;
  hiddenStates = dense->forward(hiddenStates);
  hiddenStates = torch::gelu(hiddenStates);
  // std::cout << "	BertIntermeddiate Output size: " << hiddenStates.sizes() << std::endl;
  return hiddenStates;
}


BertOutputImpl::BertOutputImpl() {}
BertOutputImpl::BertOutputImpl(Config const &config)
  : dense (torch::nn::Linear(config.intermediateSize, config.hiddenSize)),
    layerNorm (torch::nn::LayerNormOptions({config.hiddenSize}).eps(LAYER_NORM_EPS)),
    dropout (torch::nn::Dropout(config.hiddenDropoutProb)) {
  register_module("dense", dense);
  register_module("layerNorm", layerNorm);
  register_module("dropout", dropout);
}

torch::Tensor BertOutputImpl::forward(torch::Tensor hiddenStates,
                                      torch::Tensor inputTensor) {
  // std::cout << "BertOutput" << std::endl;
  hiddenStates = dense->forward(hiddenStates);
  hiddenStates = dropout->forward(hiddenStates);
  hiddenStates = layerNorm->forward(hiddenStates + inputTensor);
  return hiddenStates;
}

BertLayerImpl::BertLayerImpl() {}
BertLayerImpl::BertLayerImpl(Config const &config)
  : attention (BertAttention(config)),
    intermediate (BertIntermediate(config)),
    output (BertOutput(config)) {
  register_module("attention", attention);
  register_module("intermediate", intermediate);
  register_module("output", output);
}

torch::Tensor BertLayerImpl::forward(torch::Tensor hiddenStates,
                                     torch::Tensor attentionMask) {
  // std::cout << "BertLayer" << std::endl;
  torch::Tensor attentionOutputs = attention->forward(hiddenStates, attentionMask);
  torch::Tensor intermediateOutput = intermediate->forward(attentionOutputs);
  torch::Tensor layerOutput = output->forward(intermediateOutput, attentionOutputs);
  // std::cout << "	BertLayer Output size: " << layerOutput.sizes() << std::endl;
  return layerOutput;
}

BertEncoderImpl::BertEncoderImpl() {}
BertEncoderImpl::BertEncoderImpl(Config const &config)
  : numLayers (config.numHiddenLayers), layer (torch::nn::ModuleList()) {
	for (uint32_t i = 0; i < config.numHiddenLayers; i++) {
		layer->push_back(BertLayer(config));
	}
  register_module("layer", layer);
}

torch::Tensor BertEncoderImpl::forward(torch::Tensor hiddenStates,
                                       torch::Tensor attentionMask) {
  // std::cout << "BertEncoder" << std::endl;
  for (const auto &module : *layer) {
    hiddenStates = module->as<BertLayer>()->forward(hiddenStates, attentionMask);
  }
  return hiddenStates;
}

BertPoolerImpl::BertPoolerImpl() {}
BertPoolerImpl::BertPoolerImpl(Config const &config)
  : dense (torch::nn::Linear(config.hiddenSize, config.hiddenSize)) {
  register_module("dense", dense);
}
torch::Tensor BertPoolerImpl::forward(torch::Tensor hiddenStates) {
  // std::cout << "BertPooler" << std::endl;
  // std::cout << "Pooler hiddenStates: " << hiddenStates.sizes() << std::endl;
  hiddenStates = hiddenStates.index({torch::indexing::Slice(), 0});
  // std::cout << "	after indexing: " << hiddenStates.sizes() << std::endl;
  torch::Tensor pooledOutput = dense->forward(hiddenStates);
  pooledOutput = torch::tanh(pooledOutput);
  return pooledOutput;
}

BertModelImpl::BertModelImpl() {}
BertModelImpl::BertModelImpl(Config const &config)
  : embeddings (BertEmbeddings(config)),
    encoder (BertEncoder(config)),
    pooler (BertPooler(config)) {
  register_module("embeddings", embeddings);
  register_module("encoder", encoder);
  register_module("pooler", pooler);
}

torch::Tensor BertModelImpl::forward(torch::Tensor inputIds) {
  // std::cout << "BertModel" << std::endl;
  torch::Tensor zeros = torch::zeros_like(inputIds).to(torch::kCUDA);
  torch::Tensor ones = torch::zeros_like(inputIds).to(torch::kCUDA);
  torch::Tensor attentionMask = torch::where(inputIds != 0, ones, zeros).to(torch::kCUDA);
  torch::Tensor extendedAttentionMask = attentionMask.unsqueeze(1).unsqueeze(2);
  extendedAttentionMask = (1.0f - extendedAttentionMask) * -10000.0f;
  torch::Tensor embeddingOutput = embeddings(inputIds);
  // std::cout << "embeddingOutput: " << embeddingOutput.sizes() << std::endl;
  torch::Tensor encoderOutputs = encoder(embeddingOutput, extendedAttentionMask);
  // std::cout << "encoderOutputs: " << encoderOutputs.sizes() << std::endl;
  torch::Tensor pooledOutput = pooler(encoderOutputs);
  // std::cout << "pooledOutput: " << pooledOutput.sizes() << std::endl;
  return pooledOutput;
}

BinaryClassifierImpl::BinaryClassifierImpl() {};
BinaryClassifierImpl::BinaryClassifierImpl(Config const &config, int numLabels)
  : dense (torch::nn::Linear(config.hiddenSize, numLabels)),
    dropout(torch::nn::Dropout(config.hiddenDropoutProb)) {
  register_module("dense", dense);
  register_module("dropout", dropout);
}

torch::Tensor BinaryClassifierImpl::forward(torch::Tensor hidden) {
  return dense->forward(dropout->forward(hidden));
}
