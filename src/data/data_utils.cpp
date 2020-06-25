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

  long numRows = ids.size();

  torch::Tensor idsTensor = torch::full({numRows * MAX_SEQUENCE_LENGTH},
                                         paddingIdx,
                                         torch::TensorOptions().dtype(dtype));

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
  return idsTensor.view({numRows, MAX_SEQUENCE_LENGTH});
}

template <typename T>
torch::Tensor idsToTensor(const std::vector<T>& ids) {
  torch::ScalarType dtype;
  if (std::is_same<T, long>::value) {
    dtype = torch::kInt64;
  } else {
    dtype = torch::kFloat;
  }

  long numRows = ids.size();

	torch::Tensor idsTensor = torch::empty(numRows,
                                         torch::TensorOptions().dtype(dtype));
  T* data = idsTensor.data_ptr<T>();
  for (const auto& v : ids) {
      *data++ = v;
  }
  return idsTensor.view({numRows});
}

template <typename T>
torch::Tensor idsToTensor(const std::vector<std::vector<T>>& ids) {
  torch::ScalarType dtype;
  if (std::is_same<T, long>::value) {
    dtype = torch::kInt64;
  } else {
    dtype = torch::kFloat;
  }

  long numRows = ids.size();
  long numColumns = ids[0].size();

  torch::Tensor idsTensor = torch::empty({numRows * numColumns},
                                         torch::TensorOptions().dtype(dtype));

  // Get a pointer to tensor and copy values sequentially
  T* data = idsTensor.data_ptr<T>();
  for (int i = 0; i < ids.size(); i++) {
    if (ids[i].size() != numColumns) {
      throw std::runtime_error("Inconsistent number of columns");
    }
    for (int j = 0; j < ids[i].size(); j++) *data++ = ids[i][j];
  }
  return idsTensor.view({numRows, numColumns});
}


torch::Tensor readTextsToTensor(const std::string& textsFname,
                                const std::string& lowercaseFname,
                                const std::string& vocabFname) {
  // Initialize tokenizer
  FullTokenizer *tokenizer = new FullTokenizer(vocabFname, lowercaseFname);

  // Prepare file stream
  std::ifstream file(textsFname);
  if (!file.is_open()) {
    throw std::runtime_error(textsFname + " not found!");
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

torch::Tensor readTextsToTensor(const std::string& modelDir,
                                const std::vector<Task>& tasks,
                                const std::string& subset) {
  std::string textsFname = tasks[0].baseDir + "/" + subset + "-texts";
  std::string vocabFname = modelDir + "/vocab.txt";
  std::string lowercaseFname = modelDir + "/lowercase";
  return readTextsToTensor(textsFname, lowercaseFname, vocabFname);
}

torch::Tensor readLabelsToTensor(const std::string& labelsFname, int taskType) {
  if (((Binary & taskType) == Binary)
      || (Regression & taskType) == Regression) {
    if ((TokenLevel & taskType) == TokenLevel) {
      // Token-level, use readLabels2D and overloaded idsToTensor with
      // sepcial token ids
      float sosId = CLASSIFICATION_IGNORE_INDEX,
            eosId = CLASSIFICATION_IGNORE_INDEX,
            paddingIdx = CLASSIFICATION_IGNORE_INDEX;
      std::vector<std::vector<float>> labels =
          readLabels2D<float>(labelsFname);
      return idsToTensor(labels, sosId, eosId, paddingIdx);
    } else if ((MultiLabel & taskType) == MultiLabel) {
      // Multi-label use readLabels2D and plain idsToTensor
      std::vector<std::vector<float>> labels =
          readLabels2D<float>(labelsFname);
      return idsToTensor(labels);
    } else {
      // Sentence-level, use readLabels and plain idsToTensor
      std::vector<float> labels = readLabels<float>(labelsFname);
      return idsToTensor(labels);
    }
  } else {
    // Multiclass
    if ((TokenLevel & taskType) == TokenLevel) {
      // Token-level, use readLabels2D and overloaded idsToTensor with
      // sepcial token ids
      long sosId = CLASSIFICATION_IGNORE_INDEX,
           eosId = CLASSIFICATION_IGNORE_INDEX,
           paddingIdx = CLASSIFICATION_IGNORE_INDEX;
      std::vector<std::vector<long>> labels =
          readLabels2D<long>(labelsFname);
      return idsToTensor(labels, sosId, eosId, paddingIdx);
    } else if ((MultiLabel & taskType) == MultiLabel) {
      // Multi-label use readLabels2D and plain idsToTensor
      std::vector<std::vector<long>> labels =
          readLabels2D<long>(labelsFname);
      return idsToTensor(labels);
    } else {
      // Sentence-level, use readLabels and plain idsToTensor
      std::vector<long> labels = readLabels<long>(labelsFname);
      return idsToTensor(labels);
    }
  }
}

std::vector<torch::Tensor> readLabelsToTensor(const std::vector<Task>& tasks,
                                              const std::string& subset) {
  std::string baseFname = tasks[0].baseDir + "/" + subset + "-";
  std::vector<torch::Tensor> labelsVector;

  for (const auto& task : tasks) {
    std::string labelsFname = baseFname + task.name;
    labelsVector.push_back(readLabelsToTensor(labelsFname, task.taskType));
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
std::vector<std::vector<T>> readLabels2D(std::string fname) {
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

int detectTaskType(std::string labelsFname) {
  std::ifstream file(labelsFname);
  if (!file.is_open()) {
    throw std::runtime_error(labelsFname + " not found!");
  }

  int taskType = 0;
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
      taskType |= TokenLevel;
  } else if (nTokens[0] != 1) {
      taskType |= MultiLabel;
  }
	std::string s = data[0][0];
	size_t p;

  // Detect type of labels
  // int -> !Regression
  // float -> Regression
  (void)std::stoi(s, &p);
  if(s.size() == p) {
    // Classification, do nothing
  } else {
    (void)std::stof(s, &p);
    if(s.size() == p) {
      taskType |= Regression;
    } 
  }

  // If not Regression, detect if bninary
	if (!((Regression & taskType) == Regression)) {
    std::set<std::string> values;
    for (const auto& lineData : data) {
      for (const auto& token : lineData) {
        values.insert(token);
      }
    }
    if (values.size() == 2) {
      taskType |= Binary;
    }
  }
  return taskType;
}

void detectTaskType(Task &task) {
  std::string labelsFname = task.baseDir + "/" + "train-" + task.name;
  task.taskType = detectTaskType(labelsFname);
}
