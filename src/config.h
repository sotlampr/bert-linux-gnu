#ifndef CONFIG_H
#define CONFIG_H

#define PADDING_IDX 0  // Padding value for token ids
#define LAYER_NORM_EPS 1e-12 
#define MAX_SEQUENCE_LENGTH 100  // Pad/truncate input sequences to that
#define DELIMITER ','  // Delimiter for label files
#define SNIFF_LINES 100  // Lines to consider when detecting task type
#define CLASSIFICATION_IGNORE_INDEX -1  // Value to ignore when computing classification loss
#define MAX_GRADIENT_NORM 1.0f  // Gradient clipping value
#define WEIGHT_DECAY 1e-2f  // Adam weight decay value

// Default arguments for train
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
