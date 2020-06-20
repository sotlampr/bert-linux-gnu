#ifndef TEXT_DATASET_H
#define TEXT_DATASET_H
#include <vector>

#include <torch/types.h>
#include <torch/data.h>

#include "train/task.h"

// torch::data::Example for multiple targets
using MultiTaskExample = torch::data::Example<torch::Tensor, std::vector<torch::Tensor>>;

// Collate a vector of MultiTaskExample(s) to tensors
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

// torch::data::Dataset implementation for text inputs and multiple targets
class TextDataset : public torch::data::Dataset<TextDataset, MultiTaskExample> {
    public:
        // Initialize dataset.
        // The files are read from [tasks.baseDir]/{texts,[task.name]}-[subset]
        explicit TextDataset(const std::string& modelDir,
                             const std::vector<Task>& tasks,
                             const std::string& subset);
			  MultiTaskExample get(size_t index) override;
			  torch::optional<size_t> size() const override;

        // Get the tensor.sizes() of all labels
        std::vector<torch::IntArrayRef> getLabelSizes() const;

        // Get the class weights for imbalanced classes loss weighting
        std::vector<torch::Tensor> getClassWeights(const std::vector<Task>& tasks) const;
    private:
        torch::Tensor texts;
        std::vector<torch::Tensor> labels;
};

using TextDatasetType = torch::data::datasets::MapDataset<TextDataset,MultiTaskStack>;

using TextDataLoaderType = torch::disable_if_t<false,std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<TextDataset,MultiTaskStack>,torch::data::samplers::RandomSampler>,std::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<TextDataset,MultiTaskStack>,torch::data::samplers::RandomSampler>>>>;

// Initialize a text dataset and maps it in a MultiTaskTask
TextDatasetType getDataset(const std::string& modelDir,
                           const std::vector<Task>& tasks,
                           const std::string& subset);

#endif
