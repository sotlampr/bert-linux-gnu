#ifndef DATA_UTILS_H
#define DATA_UTILS_H
#include <vector>

#include <torch/types.h>

#include "train/task.h"


torch::Tensor readTextsToTensor(const std::string& modelDir,
                                const std::vector<Task>& tasks,
                                const std::string& subset);

std::vector<torch::Tensor> readLabelsToTensor(const std::vector<Task>& tasks,
                                              const std::string& subset);
template <typename T>
std::vector<T> readLabels(std::string fname);

template <typename T>
std::vector<std::vector<T>> readLabelsTokenLevel(std::string fname);

template <typename T> T stringToNumber(const std::string& s);

void detectTaskType(Task& task);

template <typename T>
torch::Tensor idsToTensor(const std::vector<std::vector<T>>& ids,
                          T& sosId, T& eosId, T& paddingIdx);
template <typename T>
torch::Tensor idsToTensor(const std::vector<T>& ids);
#endif
