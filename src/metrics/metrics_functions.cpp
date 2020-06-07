#include <cmath>
#include <limits>
#include "metrics_utils.h"

float matthewsCorrelationCoefficient(torch::Tensor &labels, torch::Tensor &predictions) {
  long TP = truePositives(labels, predictions);
  std::cout << "TP: " << TP << std::endl;
  long TN = trueNegatives(labels, predictions);
  std::cout << "TN: " << TN << std::endl;
  long FP = falsePositives(labels, predictions);
  std::cout << "FP: " << FP << std::endl;
  long FN = falseNegatives(labels, predictions);
  std::cout << "FN: " << FN << std::endl;

  if ((TN + FN) == 0) {
    return -std::numeric_limits<float>::infinity();
  }

  long numerator = (TP * TN) - (FP * FN);
  std::cout << "Numerator: " << numerator << std::endl;
  long denominator = std::sqrt(
    (TP + FP) * (TP + FN) * (TN + FP) *  (TN + FN)
  );
  std::cout << "Denominator: " << denominator << std::endl;
  return static_cast<float>(numerator) / static_cast<float>(denominator);
}

float accuracy(torch::Tensor &labels, torch::Tensor &predictions) {
  long TP = truePositives(labels, predictions);
  long TN = trueNegatives(labels, predictions);

  long numerator = TP + TN;
  long denominator = labels.sizes()[0];
  return static_cast<float>(numerator) / static_cast<float>(denominator);
}

float f1Score(torch::Tensor &labels, torch::Tensor &predictions) {
  long TP = truePositives(labels, predictions);
  long FP = falsePositives(labels, predictions);
  long FN = falseNegatives(labels, predictions);

  long numerator = 2 * TP;
  long denominator = numerator + FP + FN;
  return static_cast<float>(numerator) / static_cast<float>(denominator);
}
