#include "bert_model.h"

BertModelImpl::BertModelImpl() {}
BertModelImpl::BertModelImpl(Config const &config)
  : embeddings (BertEmbeddings(config)),
    encoder (BertEncoder(config)) {
  register_module("embeddings", embeddings);
  register_module("encoder", encoder);
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
  // std::cout << "pooledOutput: " << pooledOutput.sizes() << std::endl;
  return encoderOutputs;
}
