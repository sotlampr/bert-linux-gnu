#include <tuple>
#include <vector>
#include <stdexcept>
#include <torch/torch.h>
#include "data.h"
#include "model.h"

inline void innerLoop(TextDataLoaderType &loader,
                      BertModel &model,
                      BinaryClassifier &classifier,
                      torch::nn::BCEWithLogitsLoss &criterion,
                      std::vector<float> &losses,
                      torch::Tensor labels,
                      torch::Tensor predictions,
                      std::function<void (torch::Tensor)> callback) {

  int batchSize = loader->options().batch_size;
  int startIdx = 0;
  for (auto& batch : *loader) {
      auto data = batch.data.cuda();
      auto batchLabels = batch.target.cuda().to(torch::kFloat);

      torch::Tensor output = model->forward(data);
      torch::Tensor logits = classifier->forward(output).squeeze();
      torch::Tensor loss = criterion->forward(logits, batchLabels);

      losses.push_back(loss.item<float>());
      labels.index_put_({torch::indexing::Slice(startIdx, startIdx+batchLabels.sizes()[0])}, batchLabels);
      predictions.index_put_({torch::indexing::Slice(startIdx, startIdx+logits.sizes()[0])}, logits);
      startIdx += batchSize;

      #ifdef DEBUG
      std::cout << "step=" << (int)(startIdx/batchSize) << ", loss=" << losses.back() << std::endl;
      #endif

      callback(loss);
  }
}

// Training
void trainLoop(BertModel &model,
          BinaryClassifier &classifier,
          TextDataLoaderType &loader,
          torch::nn::BCEWithLogitsLoss &criterion,
          torch::optim::Optimizer &optimizer,
          std::vector<float> &losses,
          torch::Tensor &labels,
          torch::Tensor &predictions) {

  model->train();
  classifier->train();
  auto callback = [&] (torch::Tensor loss) {
      model->zero_grad();
      loss.backward();
      optimizer.step();
  };
  innerLoop(loader, model, classifier, criterion, losses, labels, predictions, callback);
}

// Validation
void trainLoop(BertModel &model,
               BinaryClassifier &classifier,
               TextDataLoaderType &loader,
               torch::nn::BCEWithLogitsLoss &criterion,
               std::vector<float> &losses,
               torch::Tensor &labels,
               torch::Tensor &predictions) {

  torch::NoGradGuard no_grad;
  model->eval();
  classifier->eval();
  auto callback = [] (torch::Tensor loss) {};
  innerLoop(loader, model, classifier, criterion, losses, labels, predictions, callback);
}
