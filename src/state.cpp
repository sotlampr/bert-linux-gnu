#include <glob.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>

#include <torch/torch.h>
#include "config.h"
#include "model.h"
#include "state.h"

std::vector<std::string> getParameterFiles (std::string path) {
    glob_t globbuf;
    std::vector<std::string> out;
    path += "/*.dat";
    int err = glob(path.c_str(), 0, NULL, &globbuf);
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

  for (auto const& fname : getParameterFiles(path)) {
    std::string paramName = getParameterName(fname);
    std::vector<int64_t> paramSize = getParameterSize(fname);
    int numValues = 1;
    for (uint32_t i = 0; i < paramSize.size(); i++) {
      numValues *= paramSize[i];
    }
    std::vector<float> values = getParameterValues(fname, numValues);
    std::cout <<  paramName << "...";

    torch::Tensor tensorValues = torch::from_blob(values.data(), paramSize);

    auto* t = modelParams.find(paramName);

    if(t != nullptr) {
      std::cout << "\tOK" << std::endl;
      t->copy_(tensorValues);
    } else {
      std::cout << "\tNot found" << std::endl;
    }
  }
}
