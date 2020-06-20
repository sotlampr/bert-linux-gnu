#include <stdexcept>
#include <string>
#include <vector>
#include "metrics.h"
#include "model.h"
#include "task.h"

Task::Task() {};

template <typename M1, typename M2>
Task::Task(Task& t,  M1 classifier_, M2 criterion_,
           LogitsToPredictionsFunc&& logitsToPredictions_)
  : name (t.name), baseDir(t.baseDir), metrics (t.metrics),
    lossMultiplier (t.lossMultiplier), taskType (t.taskType),
    criterion(criterion_), classifier(classifier_),
    logitsToPredictions (logitsToPredictions_) {};

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

// Template instatiations for the different loss options
template Task::Task<BinaryClassifier, torch::nn::BCEWithLogitsLoss >
(Task&, BinaryClassifier, torch::nn::BCEWithLogitsLoss, LogitsToPredictionsFunc&&);

template Task::Task<MulticlassClassifier, torch::nn::CrossEntropyLoss>
(Task&, MulticlassClassifier, torch::nn::CrossEntropyLoss, LogitsToPredictionsFunc&&);
