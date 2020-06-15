#ifndef CONFIG_H
#define CONFIG_H

#define PADDING_IDX 0
#define LAYER_NORM_EPS 1e-12
#define MAX_SEQUENCE_LENGTH 100
#define DELIMITER ','
#define SNIFF_LINES 100
#define CLASSIFICATION_IGNORE_INDEX -1
#define MAX_GRADIENT_NORM 1.0f
#define WEIGHT_DECAY 1e-2f
#define DEFAULT_BATCH_SIZE 32
#define DEFAULT_NUM_EPOCHS 4
#define DEFAULT_LR 1e-5f


struct Config {
    int hiddenSize;;
		float attentionDropoutProb;
		float hiddenDropoutProb;
		int intermediateSize;
    int maxPositionEmbeddings;
		int numAttentionHeads;
		int numHiddenLayers;
		int typeVocabSize;
		int vocabSize;
};
#endif
