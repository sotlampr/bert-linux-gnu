#include "metrics_functions.h"

#include <cmath>
#include <limits>

#include "metrics_utils.h"

float matthewsCorrelationCoefficient(torch::Tensor &labels, torch::Tensor &predictions) {
  // TODO: multi-class case
  long TP = truePositives(labels, predictions);
  long TN = trueNegatives(labels, predictions);
  long FP = falsePositives(labels, predictions);
  long FN = falseNegatives(labels, predictions);

  if ((TN + FN) == 0) {
    return -std::numeric_limits<float>::infinity();
  }

  long numerator = (TP * TN) - (FP * FN);
  long denominator = std::sqrt(
    (TP + FP) * (TP + FN) * (TN + FP) *  (TN + FN)
  );
  return static_cast<float>(numerator) / static_cast<float>(denominator);
}

float accuracy(torch::Tensor &labels, torch::Tensor &predictions) {
  long numerator = (labels == predictions).sum().item<long>();
  long denominator = labels.size(0);
  return static_cast<float>(numerator) / static_cast<float>(denominator);
}

float f1Score(torch::Tensor &labels, torch::Tensor &predictions) {
  // TODO: multi-class case
  long TP = truePositives(labels, predictions);
  long FP = falsePositives(labels, predictions);
  long FN = falseNegatives(labels, predictions);

  long numerator = 2 * TP;
  long denominator = numerator + FP + FN;
  return static_cast<float>(numerator) / static_cast<float>(denominator);
}
