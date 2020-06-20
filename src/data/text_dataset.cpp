#include "text_dataset.h"

#include "config.h"
#include "data_utils.h"


TextDataset::TextDataset(const std::string& modelDir,
                         const std::vector<Task>& tasks,
                         const std::string& subset)
  : texts (readTextsToTensor(modelDir, tasks, subset)),
    labels (readLabelsToTensor(tasks, subset)) {}

MultiTaskExample TextDataset::get(size_t index) {
  // Since we have multiple labels, they are collected in a vector
  std::vector<torch::Tensor> labelsOut;
  for (auto it = labels.begin(); it != labels.end(); it++) {
    labelsOut.push_back((*it)[index]);
  }
	return {texts[index], labelsOut};
}

torch::optional<size_t> TextDataset::size() const {
	return texts.size(0);
}

std::vector<torch::IntArrayRef> TextDataset::getLabelSizes() const {
  std::vector<torch::IntArrayRef> labelSizes;
  for (auto it = labels.begin(); it != labels.end(); it++) {
    labelSizes.push_back((*it).sizes());
  }
  return labelSizes;
}

std::vector<torch::Tensor> TextDataset::getClassWeights(const std::vector<Task>& tasks) const {
  std::vector<torch::Tensor> out;
  using namespace torch::indexing;
  for (size_t i = 0; i < tasks.size(); i++) {
    if ((Binary & tasks[i].taskType) == Binary) {
      // Binary task - weight is given by num_negative/num_positive
      // Values other than {0, 1} are ignored
      if ((MultiLabel & tasks[i].taskType) == MultiLabel) {
        torch::Tensor weights = torch::empty({labels[i].size(1)});
        for (int j = 0; j < labels[i].size(1); j++ ) {
          torch::Tensor pos =
              (labels[i].index({Ellipsis, j}) == 1).sum().to(torch::kFloat);
          torch::Tensor neg =
              (labels[i].index({Ellipsis, j}) == 0).sum().to(torch::kFloat);
          weights[j] = neg/pos;
        }
        out.push_back(weights.cuda());
      } else {
        torch::Tensor pos = (labels[i] == 1).sum().to(torch::kFloat);
        torch::Tensor neg = (labels[i] == 0).sum().to(torch::kFloat);
        out.push_back((neg/pos).cuda().unsqueeze(0));
      }
    } else {
      // Multiclass task.
      if ((MultiLabel & tasks[i].taskType) == MultiLabel) {
        throw std::runtime_error("Multi-label multi-class tasks not supported");
      }
      // Weight is given by num_samples / (num_classes * num_classX)
      // Binary task - weight is given by num_negative/num_positive
      torch::Tensor numClasses = (labels[i].max() + 1).to(torch::kFloat);
      torch::Tensor weights = torch::zeros(numClasses.item<long>()).to(torch::kFloat);;
      long numSamples = labels[i].size(0);

      // Token-level case
      if (labels[i].ndimension() > 1) numSamples *= labels[i].size(1);

      // Do not count ignored values in total number of sampes
      numSamples -= (labels[i] == CLASSIFICATION_IGNORE_INDEX).sum().item<long>();

      for (long j = 0; j < numClasses.item<long>(); j++) {
        weights[j] = numSamples / (numClasses * (labels[i] == j).sum());
      }
      out.push_back(weights.cuda());
    }
  }
  return out;
}

TextDatasetType getDataset(const std::string& modelDir,
                           const std::vector<Task>& tasks,
                           const std::string& subset) {
  return TextDataset(modelDir, tasks, subset).map(MultiTaskStack());
}
