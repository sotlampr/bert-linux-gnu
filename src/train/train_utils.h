#ifndef TRAIN_UTILS_H
#define TRAIN_UTILS_H
#include <string>
#include <vector>

#include "config.h"
#include "data.h"
#include "task.h"

// Initialize required objects (models, tasks, optimizer) and run training
void runTraining(const std::string& modelDir,
                 const std::string& dataDir,
                 std::vector<Task>& tasks,
                 int batchSize,
                 int numEpochs,
                 float lr,
                 int numWorkers,
                 const std::string& saveModel,
                 int randomSeed);

// Initialize "second-stage" tasks from some "first-stage" tasks.
// Adds appropriate classifier, criterion and logitsToPredictions for each task.
// If saveFname is given, saves the configurations of each classifier head
std::vector<Task> initTasks(std::vector<Task>& tasks,
                            TextDatasetType& dataset,
                            const Config& config,
                            const std::string& saveFname);
#endif
