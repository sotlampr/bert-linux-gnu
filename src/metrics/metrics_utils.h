#ifndef METRICS_UTILS_H
#define METRICS_UTILS_H
#include <torch/torch.h>
torch::Tensor logicalAnd(torch::Tensor &a, torch::Tensor &b);
long truePositives(torch::Tensor &labels, torch::Tensor &predictions);
long falsePositives(torch::Tensor &labels, torch::Tensor &predictions);
long trueNegatives(torch::Tensor &labels, torch::Tensor &predictions);
long falseNegatives(torch::Tensor &labels, torch::Tensor &predictions);
#endif
