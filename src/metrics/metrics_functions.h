#ifndef MATTHEWS_CORRELATION_H
#define MATTHEWS_CORRELATION_H
#include <torch/torch.h>
float matthewsCorrelationCoefficient(torch::Tensor &labels, torch::Tensor &predictions);
float accuracy(torch::Tensor &labels, torch::Tensor &predictions);
float f1Score(torch::Tensor &labels, torch::Tensor &predictions);
#endif
