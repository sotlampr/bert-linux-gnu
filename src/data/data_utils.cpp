#include "data_utils.h"

#include <set>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <torch/types.h>

#include "config.h"
#include "tokenize.h"


template <typename T>
torch::Tensor idsToTensor(const std::vector<std::vector<T>>& ids,
                          T& sosId, T& eosId, T& paddingIdx) {
  torch::ScalarType dtype;
  if (std::is_same<T, long>::value) {
    dtype = torch::kInt64;
  } else {
    dtype = torch::kFloat;
  }

  torch::Tensor idsTensor = torch::full({(long)ids.size()* MAX_SEQUENCE_LENGTH},
                                         paddingIdx, torch::TensorOptions().dtype(dtype));

  // Get a pointer to tensor and copy values sequentially
  T* data = idsTensor.data_ptr<T>();
  for (int i = 0; i < ids.size(); i++) {
      *data++ = sosId;
      int elementsInserted = 1;
    for (int j = 0; j < ids[i].size(); j++) {
      if (j >= MAX_SEQUENCE_LENGTH - 2) {
        std::cerr << "WARNING: truncating sequence to " << MAX_SEQUENCE_LENGTH << std::endl;
        break;
      }
      *data++ = ids[i][j];
      elementsInserted++;
    }
      *data++ = eosId;
      elementsInserted++;

    // Forward pointer to next row
    for (int k = elementsInserted; k < MAX_SEQUENCE_LENGTH; k++) data++;
  }
  idsTensor = idsTensor.view({(long)ids.size(), MAX_SEQUENCE_LENGTH});
  return idsTensor.view({(long)ids.size(), MAX_SEQUENCE_LENGTH});
}

template <typename T>
torch::Tensor idsToTensor(const std::vector<T>& ids) {
  torch::ScalarType dtype;
  if (std::is_same<T, long>::value) {
    dtype = torch::kInt64;
  } else {
    dtype = torch::kFloat;
  }

	torch::Tensor idsTensor = torch::empty(
			ids.size(), torch::TensorOptions().dtype(dtype));
  T* data = idsTensor.data_ptr<T>();
  for (const auto& v : ids) {
      *data++ = v;
  }
  return idsTensor.view({(long)ids.size()});
}


torch::Tensor readTextsToTensor(const std::string& modelDir,
                                const std::vector<Task>& tasks,
                                const std::string& subset) {
  // Initialize tokenizer
  FullTokenizer *tokenizer = new FullTokenizer(modelDir);

  // Prepare file stream
  std::string fname = tasks[0].baseDir + "/" + subset + "-texts";
  std::ifstream file(fname);
  if (!file.is_open()) {
    throw std::runtime_error(fname + " not found!");
  }

  // Read file line-by-line and tokenize to ids
  std::string line;
  std::vector<std::vector<long>> textsIds;
  while (std::getline(file, line)) {
    textsIds.push_back(tokenizer->tokenizeToIds(line));
  }

  // Convert std::vector ids to torch::Tensor
  long sosId = tokenizer->tokenToId("[CLS]"),
       eosId = tokenizer->tokenToId("[SEP]"),
       paddingIdx = PADDING_IDX;
  return idsToTensor(textsIds, sosId, eosId, paddingIdx);
}


std::vector<torch::Tensor> readLabelsToTensor(const std::vector<Task>& tasks,
                                              const std::string& subset) {
  std::string baseFname = tasks[0].baseDir + "/" + subset + "-";
  std::vector<torch::Tensor> labelsVector;

  for (const auto& task : tasks) {
    if ((TokenLevel & task.taskType) == TokenLevel) {
			if ((Binary & task.taskType) == Binary
          || (Regression & task.taskType) == Regression) {
				float sosId = CLASSIFICATION_IGNORE_INDEX,
					    eosId = CLASSIFICATION_IGNORE_INDEX,
					    paddingIdx = CLASSIFICATION_IGNORE_INDEX;
				std::vector<std::vector<float>> labels =
					 	readLabelsTokenLevel<float>(baseFname + task.name);
				labelsVector.push_back(idsToTensor(labels, sosId, eosId, paddingIdx));
			} else {
				long sosId = CLASSIFICATION_IGNORE_INDEX,
						 eosId = CLASSIFICATION_IGNORE_INDEX,
						 paddingIdx = CLASSIFICATION_IGNORE_INDEX;
				std::vector<std::vector<long>> labels =
						readLabelsTokenLevel<long>(baseFname + task.name);
				labelsVector.push_back(idsToTensor(labels, sosId, eosId, paddingIdx));
			}
		} else {
			if ((Binary & task.taskType) == Binary
          || (Regression & task.taskType) == Regression) {
				std::vector<float> labels = readLabels<float>(baseFname + task.name);
				labelsVector.push_back(idsToTensor(labels));
			} else {
				std::vector<long> labels = readLabels<long>(baseFname + task.name);
				labelsVector.push_back(idsToTensor(labels));
			}
		}
	}
  return labelsVector;
}

template <typename T> T stringToNumber(const std::string& s) {}
template<> long stringToNumber(const std::string& s) { return std::stol(s); }
template<> float stringToNumber(const std::string& s) { return std::stol(s); }

template <typename T>
std::vector<T> readLabels(std::string fname) {
  std::ifstream file(fname);
  if (!file.is_open()) {
    throw std::runtime_error(fname + " not found!");
  }
  std::string line;
  std::vector<T> out;
  while (std::getline(file, line)) {
    out.push_back(stringToNumber<T>(line));
  }
  return out;
}
template <typename T>
std::vector<std::vector<T>> readLabelsTokenLevel(std::string fname) {
  std::ifstream file(fname);
  if (!file.is_open()) {
    throw std::runtime_error(fname + " not found!");
  }
  std::string line;
	std::vector<std::vector<T>> data;
  while (std::getline(file, line)) {
		std::string value;
		std::vector<T> lineData;
		std::istringstream iss(line);
		while(std::getline(iss, value, DELIMITER)) {
			lineData.push_back(stringToNumber<T>(value));
		}
		data.push_back(lineData);
  }
  return data;
}


void detectTaskType(Task &task) {
  std::string fname = task.baseDir + "/" + "train-" + task.name;
  std::ifstream file(fname);
  if (!file.is_open()) {
    throw std::runtime_error(fname + " not found!");
  }
  std::string line;
	std::vector<int> nTokens;
	std::vector<std::vector<std::string>> data;

	// Read some lines for sniffing
	for (int i = 0; i < SNIFF_LINES; i++) {
		file >> line;
		std::string value;
		std::vector<std::string> lineData;
		std::istringstream iss(line);
		while(std::getline(iss, value, DELIMITER)) {
			lineData.push_back(value);
		}
		data.push_back(lineData);
		nTokens.push_back(lineData.size());
	}

  // Detect if lines have different numbers of tokens
	if (std::adjacent_find(nTokens.begin(), nTokens.end(), std::not_equal_to<int>()) != nTokens.end()) {
      task.taskType |= TokenLevel;
  }
	std::string s = data[0][0];
	size_t p;

  // Detect type of labels
  // int -> !Regression
  // float -> Regression
  // string -> !Regression w/ NeedsTranslation
  try {
		(void)std::stoi(s, &p);
		if(s.size() == p) {
      // Classification, do nothing
		} else {
			(void)std::stof(s, &p);
			if(s.size() == p) {
        task.taskType |= Regression;
			} 
		}
	} catch (std::invalid_argument& e) {
    task.taskType |= NeedsTranslation;
	}

  // If not Regression, detect if bninary
	if (!((Regression & task.taskType) == Regression)) {
    std::set<std::string> values;
    for (const auto& lineData : data) {
      for (const auto& token : lineData) {
        values.insert(token);
      }
    }
    if (values.size() == 2) {
      task.taskType |= Binary;
    }
  }
}
