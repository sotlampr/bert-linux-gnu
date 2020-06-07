#include <stdexcept>
#include <string>
#include <vector>
#include "task.h"

Task::Task() {};

void Task::init(const std::string &taskName) {
  if (!name.empty()) {
    throw std::runtime_error("Task `" + taskName + "` already initialized!");
  }
  name = taskName;
}

std::string Task::getName() const {
  return name;
}

void Task::addMetric(const std::string &metric) {
  metrics.push_back(metric);
}

std::vector<std::string> Task::getMetrics() const {
  return metrics;
};

void Task::setLossMultiplier(float taskLossMultiplier) {
  lossMultiplier = taskLossMultiplier;
}

float Task::getLossMultiplier() const {
  return lossMultiplier;
}

template <typename M>
void Task::setCriterion(M taskCriterion) {
  criterion<M> = taskCriterion;
}

template <typename M>
M Task::getCriterion() const {
  return criterion<M>;
}

template <typename M> M Task::criterion;
template torch::nn::BCEWithLogitsLoss Task::criterion<torch::nn::BCEWithLogitsLoss>;
template void Task::setCriterion<torch::nn::BCEWithLogitsLoss>(torch::nn::BCEWithLogitsLoss);
template torch::nn::BCEWithLogitsLoss Task::getCriterion<torch::nn::BCEWithLogitsLoss>() const;
