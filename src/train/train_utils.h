#ifndef TRAIN_UTILS_H
#define TRAIN_UTILS_H
#include <string>
#include <vector>

#include "config.h"
#include "data.h"
#include "task.h"

void runTraining(const Config &config,
                 const std::string &modelDir,
                 const std::string &dataDir,
                 std::vector<Task> &tasks,
                 int batchSize,
                 int numWorkers,
                 int numEpochs);

std::vector<Task> initTasks(std::vector<Task>& tasks,
                            TextDatasetType& dataset,
                            const Config& config);
#endif
