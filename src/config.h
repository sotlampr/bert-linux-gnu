#ifndef CONFIG_H
#define CONFIG_H

#include <stddef.h>
#define PADDING_IDX 0
#define LAYER_NORM_EPS 1e-12
#define MAX_SEQUENCE_LENGTH 100
#define DELIMITER ','
#define SNIFF_LINES 100
#define DO_LOWERCASE true  // TODO: detect automatically based on model

class Config {
  public:
    int hiddenSize = 768;;
		float attentionDropoutProb = 0.1f;
		float hiddenDropoutProb = 0.1f;
		int intermediateSize = 3072;
    int maxPositionEmbeddings = 512;
		int numAttentionHeads = 12;
		int numHiddenLayers = 12;
		int typeVocabSize = 2;
		int vocabSize = 30522;
};
#endif
