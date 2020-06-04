#include <torch/torch.h>
#include <fstream>
#include <string>
#include <vector>

#include "config.h"
#include "data.h"

std::vector<std::string> readTextFile(std::string fname) {
  std::ifstream file(fname);
  std::string line;
  std::vector<std::string> out;
  while (std::getline(file, line)) {
    out.push_back(line);
  }
  return out;
}

std::vector<long> readClassificationLabels(std::string fname) {
  std::ifstream file(fname);
  std::string line;
  std::vector<long> out;
  while (std::getline(file, line)) {
    out.push_back(std::stol(line));
  }
  return out;
}

torch::Tensor idsToTensor(std::vector<std::vector<long>> const &v,
                          long sosId, long eosId) {
  torch::Tensor out = torch::full({(long)v.size()* MAX_SEQUENCE_LENGTH}, PADDING_IDX, 
                                  torch::TensorOptions().dtype(torch::kInt64));
  long* data = out.data_ptr<long>();
  for (int i = 0; i < v.size(); i++) {
      *data++ = sosId;
      int elementsInserted = 1;
    for (int j = 0; j < v[i].size(); j++) {
      if (j >= MAX_SEQUENCE_LENGTH - 2) {
        std::cerr << "WARNING: truncating sequence to " << MAX_SEQUENCE_LENGTH << std::endl;
        break;
      }
      *data++ = v[i][j];
      elementsInserted++;
    }
      *data++ = eosId;
      elementsInserted++;
    for (int k = elementsInserted; k < MAX_SEQUENCE_LENGTH; k++) data++;
  }
  return out.view({(long)v.size(), MAX_SEQUENCE_LENGTH});
}

torch::Tensor labelsToTensor(std::vector<long> const &v) {
  torch::Tensor out = torch::empty(v.size(), torch::TensorOptions().dtype(torch::kInt64));
  long* data = out.data_ptr<long>();
  for (const auto& i : v) {
      *data++ = i;
  }
  return out.view({(long)v.size()});
}

TextDataset::TextDataset(std::vector<std::vector<long>> const &texts,
                         std::vector<long> const &labels,
                         long sosId, long eosId)
	: texts (idsToTensor(texts, sosId, eosId)), labels (labelsToTensor(labels)) {};

torch::data::Example<> TextDataset::get(size_t index) {
	return {texts[index], labels[index]};
}

torch::optional<size_t> TextDataset::size() const {
	return texts.size(0);
}
