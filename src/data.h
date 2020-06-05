#ifndef DATA_H
#define DATA_H
#include <fstream>
#include <string>
#include <vector>

#include <torch/torch.h>

enum Task {
  Regression = 1 << 0,
  Binary     = 1 << 1,
  Multilabel = 1 << 2,
  Sequence   = 1 << 3
};

std::vector<std::string> readTextFile(std::string fname);
std::vector<long>  readClassificationLabels(std::string fname);
torch::Tensor idsToTensor(std::vector<std::vector<long>> const &v,
                          long sosId, long eosId);
torch::Tensor labelsToTensor(std::vector<long> const &v);

class TextDataset : public torch::data::Dataset<TextDataset> {
    public:
        explicit TextDataset(std::vector<std::vector<long>> const &texts,
                             std::vector<long> const &labels,
                             long sosId, long eosId);
			  torch::data::Example<> get(size_t index) override;
			  torch::optional<size_t> size() const override;
    private:
        torch::Tensor texts, labels;
};

torch::data::datasets::MapDataset<TextDataset, torch::data::transforms::Stack<torch::data::Example<>>>
  readFileToDataset(std::string const &modelPath, bool doLowercase, std::string const &dataPrefix);
#endif
