#include <stdexcept>
#include <string>
#include <vector>
#include "task.h"
#include "metrics.h"

Task::Task() {};

void Task::addMetric(std::string metric) {
  if (metric == "accuracy") {
    metrics.push_back(std::make_pair(metric, accuracy));
    return;
  }
  if (metric == "f1") {
    metrics.push_back(std::make_pair(metric, f1Score));
    return;
  }
  if (metric == "matthewscc") {
    metrics.push_back(std::make_pair(metric, matthewsCorrelationCoefficient));
    return;
  }
  throw std::runtime_error("No metric `" + metric + "`");
}

template <typename M> M Task::criterion;
template torch::nn::BCEWithLogitsLoss Task::criterion<torch::nn::BCEWithLogitsLoss>;
