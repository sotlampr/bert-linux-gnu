#include <torch/torch.h>

#include "config.h"
#include "bert_encoder.h"
#include "bert_embeddings.h"
#include "bert_model.h"
#include "bert_pooler.h"

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
