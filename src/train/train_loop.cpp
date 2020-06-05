#include <tuple>
#include <vector>
#include <stdexcept>
#include <torch/torch.h>
#include "data.h"
#include "model.h"

inline torch::Tensor doStep(torch::data::Example<> &batch,
                            BertModel &model,
                            BinaryClassifier &classifier,
                            torch::nn::BCEWithLogitsLoss &criterion,
                            std::vector<torch::Tensor> &predictions,
                            std::vector<float> &losses) {

      auto data = batch.data.cuda();
      auto labels = batch.target.cuda().to(torch::kFloat);

      torch::Tensor output = model->forward(data);
      torch::Tensor logits = classifier->forward(output).squeeze();
      predictions.push_back(logits);

      torch::Tensor loss = criterion->forward(logits, labels);
      losses.push_back(loss.item<float>());
      return loss;
}


std::tuple<std::vector<float>, std::vector<torch::Tensor>>
trainLoop(BertModel &model,
          BinaryClassifier &classifier,
          TextDataLoaderType &loader,
          torch::nn::BCEWithLogitsLoss &criterion,
          torch::optim::Optimizer &optimizer) {

  std::vector<float> losses;
  std::vector<torch::Tensor> predictions;

  model->train(true);

  int step = 1;
  for (auto batch : *loader) {
      torch::Tensor loss = doStep(batch, model, classifier, criterion, predictions, losses);
      #ifdef DEBUG
      std::cout << "step=" << step << ", loss=" << losses.back() << std::endl;
      #endif
      model->zero_grad();
      loss.backward();
      optimizer.step();
      step++;
  }
  return {losses, predictions};
}

std::tuple<std::vector<float>, std::vector<torch::Tensor>>
trainLoop(BertModel &model,
          BinaryClassifier &classifier,
          TextDataLoaderType &loader,
          torch::nn::BCEWithLogitsLoss &criterion) {

  torch::NoGradGuard no_grad;
  std::vector<float> losses;
  std::vector<torch::Tensor> predictions;

  model->train(false);

  int step = 1;
  for (auto& batch : *loader) {
      doStep(batch, model, classifier, criterion, predictions, losses);
      #ifdef DEBUG
      std::cout << "step=" << step << ", loss=" << losses.back() << std::endl;
      #endif
      step++;
  }
  return {losses, predictions};
}
