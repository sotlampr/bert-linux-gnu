#ifndef TASK_H
#define TASK_H
#include <string>
#include <vector>

#include <torch/nn/modules/container/any.h>
#include <torch/nn/modules/loss.h>
#include <torch/types.h>

// Type for a metric, tuple of (name, metric_function)
using Metric = std::pair<std::string,std::function<float (torch::Tensor&, torch::Tensor&)>>;

// Function that receives a tensor of logits and returns the predicted labels
using LogitsToPredictionsFunc = std::function<torch::Tensor (torch::Tensor&)>;

// Bit mask for the task type options
enum TaskType {
  Regression       = 1 << 0,
  Binary           = 1 << 1,
  TokenLevel       = 1 << 2,
  NeedsTranslation = 1 << 3,
};

// Represents a task.
// Should be initialized in two stages:
//   First stage:
//     Initialize with empty constructor and set task name, directory,
//     metrics, loss multiplier
//   Second stage:
//     Initialize using the template constructor given appropriate
//     classifier, criterion and LogitsToPrediction items
class Task {
  public:
    // Construct with classifier, criterion and logitsToPredictions function
    template <typename M1, typename M2>
    Task(Task& t,  M1 classifier_, M2 criterion_,
         LogitsToPredictionsFunc&& logitsToPredictions_);

    Task();

    void addMetric(std::string metric); // Add a metric from a string description
    std::string name;  // The name of the task
    std::string baseDir;  // Directory where the task files are found
    std::vector<Metric> metrics;  
    float lossMultiplier = 0.1f;
    int taskType = 0;
    torch::nn::AnyModule criterion;
    torch::nn::AnyModule classifier;
    LogitsToPredictionsFunc logitsToPredictions;
};
#endif
