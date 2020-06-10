#ifndef STATE_UTILS_H
#define STATE_UTILS_H
#include <string>
#include <vector>

#include <torch/nn/module.h>

#include "config.h"

std::vector<std::string> getParameterFiles (std::string path);
std::string getParameterName(std::string fname);
std::vector<int64_t> getParameterSize(std::string fname);
std::vector<float> getParameterValues(std::string fname, int expectedSize);
void loadState(const std::string &path, torch::nn::Module& model);
#endif 
