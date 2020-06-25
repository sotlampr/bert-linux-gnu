#ifndef DATA_UTILS_H
#define DATA_UTILS_H
#include <vector>

#include <torch/types.h>

#include "train/task.h"

// Read a text file and return a padded tensor of embedding indices
torch::Tensor readTextsToTensor(const std::string& textsFname,
                                const std::string& vocabFname,
                                const std::string& lowercaseFname);

torch::Tensor readTextsToTensor(const std::string& modelDir,
                                const std::vector<Task>& tasks,
                                const std::string& subset);

// Read labels for the given task into a vector of tensors of indices
torch::Tensor readLabelsToTensor(const std::string& labelsFname, int taskType);
std::vector<torch::Tensor> readLabelsToTensor(const std::vector<Task>& tasks,
                                              const std::string& subset);

// Read labels from a filename to a vector of C++ types
// Sentence-level: one label per line
template <typename T>
std::vector<T> readLabels(std::string fname);

// Read labels from a filename to a vector of vectors of C++ types
// 2D: multiple labels per line
template <typename T>
std::vector<std::vector<T>> readLabels2D(std::string fname);

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

// Convert a vector of C++ types to a torch tensor
// For multi-label tasks
template <typename T>
torch::Tensor idsToTensor(const std::vector<std::vector<T>>& ids);


// Detect the task type by sniffing the first lines of a file
int detectTaskType(std::string labelsFname);
void detectTaskType(Task& task);
#endif
