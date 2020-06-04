#ifndef CONFIG_H
#define CONFIG_H

#define PADDING_IDX 0
#define LAYER_NORM_EPS 1e-12
#define MAX_SEQUENCE_LENGTH 75

class Config {
  public:
    uint32_t hiddenSize = 768;;
		float attentionDropoutProb = 0.1f;
		float hiddenDropoutProb = 0.1f;
		uint32_t intermediateSize = 3072;
    uint32_t maxPositionEmbeddings = 512;
		uint32_t numAttentionHeads = 12;
		uint32_t numHiddenLayers = 12;
		uint32_t typeVocabSize = 2;
		uint32_t vocabSize = 30522;
};
#endif
