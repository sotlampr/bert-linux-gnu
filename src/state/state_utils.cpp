#include "state_utils.h"

#include <glob.h>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include <cassert>

#include <torch/utils.h>

#include "model.h"

std::vector<std::string> getGlobFiles (const std::string& pattern) {
    glob_t globbuf;
    std::vector<std::string> out;
    int err = glob(pattern.c_str(), 0, NULL, &globbuf);
    if(err == 0)
    {
        for (size_t i = 0; i < globbuf.gl_pathc; i++) {
          out.push_back(globbuf.gl_pathv[i]);
        }
    }
    globfree(&globbuf);
    return out;
}

std::string getParameterName(std::string fname) {
  size_t from = fname.find_last_of("\\/") + 1;
  size_t to = fname.find_last_of("-");
  return fname.substr(from, to-from);
}

std::vector<int64_t> getParameterSize(std::string fname) {
  size_t from = fname.find_last_of("-") + 1;
  size_t to = fname.find_last_of(".dat");
  std::vector<int64_t> size;
  std::string sizeString = fname.substr(from, to-from);
  size_t loc;
  while ((loc = sizeString.find("_")) != std::string::npos) {
    size.push_back(std::stoi(sizeString.substr(0, loc)));
    sizeString.erase(0, loc+1);
  }
  size.push_back(std::stoi(sizeString));
  return size;
}

std::vector<float> getParameterValues(std::string fname, int expectedSize) {
  std::ifstream ifs;
  ifs.open(fname, std::ifstream::in | std::ifstream::binary);
  int i = 0;
  std::vector<float> out;
  out.reserve(expectedSize);
  float f;
  while (ifs.read(reinterpret_cast<char*>(&f), sizeof(float))) {
    out[i] = f;
    i++;
  }
  assert (i == expectedSize);
  return out;
}

void loadState(const std::string &path, torch::nn::Module& model) {
  torch::NoGradGuard no_grad;
  auto modelParams = model.named_parameters(true /*recurse*/);
  auto modelBuffers = model.named_buffers(true /*recurse*/);

  #ifdef DEBUG
  std::cout << "Loading state from `" << path << "`" << std::endl;
  #endif

  for (auto const& fname : getGlobFiles(path + "/*.dat")) {
    std::string paramName = getParameterName(fname);
    std::vector<int64_t> paramSize = getParameterSize(fname);
    int numValues = 1;
    for (size_t i = 0; i < paramSize.size(); i++) {
      numValues *= paramSize[i];
    }
    std::vector<float> values = getParameterValues(fname, numValues);

    #ifdef DEBUG
    std::cout << "\t->" <<  paramName << "...";
    #endif

    torch::Tensor tensorValues = torch::from_blob(values.data(), paramSize);

    auto* t = modelParams.find(paramName);

    if(t != nullptr) {
      #ifdef DEBUG
      std::cout << "\tOK" << std::endl;
      #endif
      t->copy_(tensorValues);
    } else {
      #ifdef DEBUG
      std::cout << "\tNot found" << std::endl;
      #endif
      std::cerr << "WARNING: Parameter " << paramName << " not in model" << std::endl;
      // throw std::runtime_error("Parameter " + paramName + " not in model");
    }
  }
}

template <typename T>
void saveStruct(const T& obj, const std::string& fname) {
  std::ofstream fs(fname, std::ios::binary);
  fs.write((char*)&obj, sizeof(T));
  fs.close();
}
template <typename T>
void readStruct(T& obj, const std::string& fname) {
  std::ifstream fs(fname, std::ios::binary);
  fs.read((char*)&obj, sizeof(T));
  fs.close();
}

// Explicit instatiations
template void saveStruct(const Config&, const std::string&);
template void saveStruct(const BinaryClassifierOptions&, const std::string&);
template void saveStruct(const MutliclassClassifierOptions&, const std::string&);
template void readStruct(Config&, const std::string&);
template void readStruct(BinaryClassifierOptions&, const std::string&);
template void readStruct(MutliclassClassifierOptions&, const std::string&);
