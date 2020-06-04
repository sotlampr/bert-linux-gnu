#ifndef STATE_H
#define STATE_H
#include <string>
#include <torch/torch.h>

void loadState(const std::string &path, torch::nn::Module& model);

#endif 
