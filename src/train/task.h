#ifndef TASK_H
#define TASK_H
#include <string>
#include <vector>

#include <torch/nn/modules/container/any.h>
#include <torch/nn/modules/loss.h>
#include <torch/types.h>

typedef std::pair<std::string,std::function<float (torch::Tensor&, torch::Tensor&)>>
Metric;

enum TaskType {
  Regression       = 1 << 0,
  Binary           = 2 << 0,
  TokenLevel       = 3 << 0,
  NeedsTranslation = 4 << 0,
};

class Task {
  public:
    // Construct with classifier, criterion and logitsToPredictions function
    template <typename M1, typename M2>
    Task(Task& t,  M1 classifier_, M2 criterion_,
         std::function<torch::Tensor (torch::Tensor)> logitsToPredictions_);

    Task();

    void addMetric(std::string metric);
    std::string name;
    std::string baseDir;
    std::vector<Metric> metrics;
    float lossMultiplier = 0.1f;
    int taskType = 0;
    torch::nn::AnyModule criterion;
    torch::nn::AnyModule classifier;
    std::function<torch::Tensor (torch::Tensor)> logitsToPredictions;
};
#endif
