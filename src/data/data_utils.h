#ifndef DATA_UTILS_H
#define DATA_UTILS_H
#include <vector>

#include <torch/types.h>

#include "train/task.h"

// Read a text file and return a padded tensor of embedding indices
torch::Tensor readTextsToTensor(const std::string& modelDir,
                                const std::vector<Task>& tasks,
                                const std::string& subset);

// Read labels for the given task into a vector of tensors of indices
std::vector<torch::Tensor> readLabelsToTensor(const std::vector<Task>& tasks,
                                              const std::string& subset);

// Read labels from a filename to a vector of C++ types
// Sentence-level: one label per line
template <typename T>
std::vector<T> readLabels(std::string fname);

// Read labels from a filename to a vector of vectors of C++ types
// Token-level: multiple labels per line
template <typename T>
std::vector<std::vector<T>> readLabelsTokenLevel(std::string fname);

// Convert string to numeric type (stol, stof)
template <typename T> T stringToNumber(const std::string& s);

// Convert a vector of C++ types to a torch tensor
template <typename T>
torch::Tensor idsToTensor(const std::vector<T>& ids);

// Convert a vector of vectors of C++ types to a padded torch tensor
// For sentence-level labels
template <typename T>
torch::Tensor idsToTensor(const std::vector<std::vector<T>>& ids,
                          T& sosId, T& eosId, T& paddingIdx);

// Detect the task type by sniffing the first lines of a file
void detectTaskType(Task& task);
#endif
