#ifndef TASK_H
#define TASK_H
#include <string>
#include <variant>
#include <vector>
#include <torch/torch.h>

typedef std::pair<std::string,std::function<float (torch::Tensor&, torch::Tensor&)>> Metric;

enum TaskType {
  Regression       = 1 << 0,
  Binary           = 2 << 0,
  TokenLevel       = 3 << 0,
  NeedsTranslation = 4 << 0,
};

class Task {
  public:
    Task();
    void addMetric(std::string metric);
    std::string name;
    std::string baseDir;
    std::vector<Metric> metrics;
    float lossMultiplier = 0.1f;
    int taskType = 0;
};

class TaskWithCriterion : public Task {
    public:
      template <typename ModuleType>
      TaskWithCriterion(Task& t,  ModuleType module);
      torch::nn::AnyModule criterion;
};
#endif
