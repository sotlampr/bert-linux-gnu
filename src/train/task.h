#ifndef TASK_H
#define TASK_H
#include <string>
#include <variant>
#include <vector>
#include <torch/torch.h>

class Task {
  public:
    Task();
    void init(const std::string &name);
    std::string getName() const;
    void addMetric(const std::string &metric);
    std::vector<std::string> getMetrics() const;
    void setLossMultiplier(float lossMultiplier);
    float getLossMultiplier() const;
    template <typename M> void setCriterion(M taskCriterion);
    template <typename M> M getCriterion() const;
  private:
    template <typename M> static M criterion;
    std::string name;
    std::vector<std::string> metrics;
    float lossMultiplier = 0.1f;
};
#endif
