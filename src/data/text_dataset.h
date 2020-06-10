#ifndef TEXT_DATASET_H
#define TEXT_DATASET_H
#include <vector>

#include <torch/types.h>
#include <torch/data.h>

#include "train/task.h"

typedef torch::data::Example<torch::Tensor, std::vector<torch::Tensor>> MultiTaskExample;

struct MultiTaskStack : public torch::data::transforms::Collation<MultiTaskExample> {
  MultiTaskExample apply_batch(std::vector<MultiTaskExample> examples) override {
    std::vector<torch::Tensor> data;
    std::vector<std::vector<torch::Tensor>> targets;
    std::vector<torch::Tensor> targetsStacked;

    targets.resize(examples[0].target.size());

    data.reserve(examples.size());

    for (auto& example : examples) {
      data.push_back(std::move(example.data));
      for (size_t i = 0; i < example.target.size(); i++) {
        targets[i].push_back(std::move(example.target[i]));
      }
    }

    for (auto it = targets.begin(); it != targets.end(); it++) {
      targetsStacked.push_back(torch::stack(*it));
    }
    return {torch::stack(data), targetsStacked};
  }
};

class TextDataset : public torch::data::Dataset<TextDataset, MultiTaskExample> {
    public:
        explicit TextDataset(const std::string& modelDir,
                             const std::vector<Task>& tasks,
                             const std::string& subset);
			  MultiTaskExample get(size_t index) override;
			  torch::optional<size_t> size() const override;
        std::vector<torch::IntArrayRef> getLabelSizes() const;
        std::vector<torch::Tensor> getClassWeights(const std::vector<Task>& tasks) const;
    private:
        torch::Tensor texts;
        std::vector<torch::Tensor> labels;
};

typedef torch::data::datasets::MapDataset<TextDataset,MultiTaskStack>
TextDatasetType;

typedef torch::disable_if_t<false,std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<TextDataset,MultiTaskStack>,torch::data::samplers::RandomSampler>,std::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<TextDataset,MultiTaskStack>,torch::data::samplers::RandomSampler>>>>
TextDataLoaderType;

TextDatasetType getDataset(const std::string& modelDir,
                           const std::vector<Task>& tasks,
                           const std::string& subset);
#endif
