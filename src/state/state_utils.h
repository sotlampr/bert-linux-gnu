#ifndef STATE_UTILS_H
#define STATE_UTILS_H
#include <string>
#include <vector>

#include <torch/nn/module.h>

#include "config.h"

// Unix-style pattern file retrieval
std::vector<std::string> getGlobFiles (const std::string& pattern);

// Get the bert parameter name from a saved binary file's filename
std::string getParameterName(std::string fname);

// Get the bert parameter size from a saved binary file's filename
std::vector<int64_t> getParameterSize(std::string fname);

// Read the bert parameters to a vector of floats from a saved binary file
std::vector<float> getParameterValues(std::string fname, int expectedSize);

// Load state from a pytorch-transformer exported model to a BertModel
void loadState(const std::string &path, torch::nn::Module& model);

// Save a struct to disk (used for saving Config(s))
template <typename T>
void saveStruct(const T& obj, const std::string& fname);

// Load a struct from disk (used for saving Config(s))
template <typename T>
void readStruct(T& obj, const std::string& fname);

#endif 
