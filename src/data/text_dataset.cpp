#include "text_dataset.h"

#include "data_utils.h"


TextDataset::TextDataset(const std::string& modelDir,
                         const std::vector<Task>& tasks,
                         const std::string& subset)
  : texts (readTextsToTensor(modelDir, tasks, subset)),
    labels (readLabelsToTensor(tasks, subset)) {}

MultiTaskExample TextDataset::get(size_t index) {
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
  for (size_t i = 0; i < tasks.size(); i++) {
    if ((TokenLevel & tasks[i].taskType) == TokenLevel) {
      throw std::runtime_error("Token-level classification not implemented");
    } else {
      if ((Binary & tasks[i].taskType) == Binary) {
        if (labels[i].ndimension() != 1) {
          throw std::runtime_error("Multilabel classification not implemented");
        } else {
          torch::Tensor pos = (labels[i] == 1).sum().to(torch::kFloat);
          torch::Tensor neg = (labels[i] == 0).sum().to(torch::kFloat);
          out.push_back((neg/pos).cuda());
        }
      } else {
        if (labels[i].ndimension() != 1) {
          throw std::runtime_error("Multilabel classification not implemented");
        } else {
          torch::Tensor numClasses = (labels[i].max() + 1).to(torch::kFloat);
          torch::Tensor weights = torch::zeros(numClasses.item<long>()).to(torch::kFloat);;
          long numSamples = labels[i].sizes()[0];

          for (long j = 0; j < numClasses.item<long>(); j++) {
            weights[j] = numSamples / (numClasses * (labels[i] == j).sum());
          }
          out.push_back(weights.cuda());
        }
      }
    }
  }
  return out;
}

TextDatasetType getDataset(const std::string& modelDir,
                           const std::vector<Task>& tasks,
                           const std::string& subset) {
  return TextDataset(modelDir, tasks, subset).map(MultiTaskStack());
}
