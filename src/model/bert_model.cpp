#include "bert_model.h"

BertModelImpl::BertModelImpl() {}
BertModelImpl::BertModelImpl(Config const &config)
  : embeddings (BertEmbeddings(config)),
    encoder (BertEncoder(config)) {
  register_module("embeddings", embeddings);
  register_module("encoder", encoder);
}

torch::Tensor BertModelImpl::forward(torch::Tensor inputIds) {
  // inputIds shape: (BATCH_SIZE, MAX_SEQUENCE_LENGTH) (non-embedded ids)

  // The attention mask is going to be added to the raw scores before the
  // softmax, so we will subtract 10,000 from the embedding for  inputs that
  // are padded
  torch::Tensor attentionMask = torch::where(
    inputIds == PADDING_IDX,
    torch::full_like(inputIds, -10000.0f), // To ignore
    torch::full_like(inputIds, 0.0f) // To attend
  ).cuda(); // shape: (BATCH_SIZE, MAX_SEQUENCE_LENGTH)

  // Convert attentionMask to (BATCH_SIZE, 1, 1, MAX_SEQUENCE_LENGTH)
  attentionMask  = attentionMask.unsqueeze(1).unsqueeze(2);

  // shapes: (BATCH_SIZE, MAX_SEQUENCE_LENGTH, HIDDEN_SIZE) (embedded ids)
  torch::Tensor embeddingOutput = embeddings(inputIds);
  torch::Tensor encoderOutputs = encoder(embeddingOutput, attentionMask);
  return encoderOutputs;
}
