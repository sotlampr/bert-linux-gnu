#ifndef DATA_UTILS_H
#define DATA_UTILS_H
#include <vector>
#include <torch/torch.h>
#include "train/task.h"

torch::Tensor readTextsToTensor(const std::string& modelDir,
                                const std::vector<Task>& tasks,
                                const std::string& subset);

std::vector<torch::Tensor> readLabelsToTensor(const std::vector<Task>& tasks,
                                              const std::string& subset);
template <typename T>
std::vector<T> readLabels(std::string fname);
void detectTaskType(Task& task);
#endif
