#include <set>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <torch/torch.h>

#include "config.h"
#include "data_utils.h"
#include "train/task.h"
#include "tokenize.h"


torch::Tensor readTextsToTensor(const std::string& modelDir,
                                const std::vector<Task>& tasks,
                                const std::string& subset) {
  // Initialize tokenizer
  FullTokenizer *tokenizer = new FullTokenizer(modelDir + "/vocab.txt", DO_LOWERCASE);

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
  long sosId = tokenizer->tokenToId("[CLS]"), eosId = tokenizer->tokenToId("[SEP]");

  torch::Tensor textsTensor = torch::full({(long)textsIds.size()* MAX_SEQUENCE_LENGTH},
                                          PADDING_IDX, torch::TensorOptions().dtype(torch::kInt64));

  // Get a pointer to tensor and copy values sequentially
  long* data = textsTensor.data_ptr<long>();
  for (int i = 0; i < textsIds.size(); i++) {
      *data++ = sosId;
      int elementsInserted = 1;
    for (int j = 0; j < textsIds[i].size(); j++) {
      if (j >= MAX_SEQUENCE_LENGTH - 2) {
        std::cerr << "WARNING: truncating sequence to " << MAX_SEQUENCE_LENGTH << std::endl;
        break;
      }
      *data++ = textsIds[i][j];
      elementsInserted++;
    }
      *data++ = eosId;
      elementsInserted++;

    // Forward pointer to next row
    for (int k = elementsInserted; k < MAX_SEQUENCE_LENGTH; k++) data++;
  }
  return textsTensor.view({(long)textsIds.size(), MAX_SEQUENCE_LENGTH});
}


std::vector<torch::Tensor> readLabelsToTensor(const std::vector<Task>& tasks,
                                              const std::string& subset) {
  std::string baseFname = tasks[0].baseDir + "/" + subset + "-";
  std::vector<torch::Tensor> labelsVector;

  for (const auto& task : tasks) {
    if ((TokenLevel & task.taskType) == TokenLevel) {
      throw std::runtime_error("Token-level classification not implemented");
    } else {
      if ((Binary & task.taskType) == Binary) {
        // Binary classification: labels<float>
        std::vector<float> labels = readLabels<float>(baseFname + task.name);

        // Convert std::vector to torch::Tensor
        torch::Tensor labelsTensor = torch::empty(labels.size(),
                                                  torch::TensorOptions().dtype(torch::kFloat));
        float* data = labelsTensor.data_ptr<float>();
        for (const auto& v : labels) {
            *data++ = v;
        }
        labelsVector.push_back(labelsTensor.view({(long)labels.size()}));
      } else {
        // Multiclass classification: labels<long>
        std::vector<long> labels = readLabels<long>(baseFname + task.name);
        torch::Tensor labelsTensor = torch::empty(labels.size(),
                                                  torch::TensorOptions().dtype(torch::kInt64));
        long* data = labelsTensor.data_ptr<long>();
        for (const auto& v : labels) {
            *data++ = v;
        }
        labelsVector.push_back(labelsTensor.view({(long)labels.size()}));
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
