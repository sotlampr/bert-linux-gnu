#ifndef TRAIN_UTILS_H
#define TRAIN_UTILS_H
#include <stddef.h>
#include <string>
#include <torch/torch.h>
#include "config.h"

void runTraining(const Config &config,
                 const std::string &modelDir, const std::string &dataDir,
                 size_t batchSize, size_t numWorkers, size_t numEpochs);
#endif
