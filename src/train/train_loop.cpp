#include <tuple>
#include <vector>
#include <stdexcept>
#include <torch/torch.h>
#include "data.h"
#include "model.h"


void innerLoop(BertModel &model,
               BinaryClassifier &classifier,
               TextDataLoaderType &loader,
               std::vector<TaskWithCriterion> &tasks,
               std::vector<std::vector<float>> &losses,
               std::vector<torch::Tensor> &labels,
               std::vector<torch::Tensor> &predictions,
               std::function<void (torch::Tensor)> callback) {

  int batchSize = loader->options().batch_size;
  int startIdx = 0;
  for (auto& batch : *loader) {
      auto data = batch.data.cuda();
      auto batchLabels = batch.target;
			std::for_each(batchLabels.begin(), batchLabels.end(), 
				[](torch::Tensor &t) {
					t = t.cuda().to(torch::kFloat);
			});

      torch::Tensor output = model->forward(data);
      torch::Tensor loss = torch::zeros(tasks.size(), torch::requires_grad()).to(torch::kCUDA);
      torch::Tensor taskLoss, taskLogits;
      
      for (size_t i = 0; i < tasks.size(); i++) {
        taskLogits = classifier->forward(output).squeeze();
        taskLoss = tasks[i].criterion.forward(taskLogits, batchLabels[i]);
        loss += taskLoss * tasks[i].lossMultiplier;
        losses[i].push_back(taskLoss.item<float>());
        labels[i].index_put_({torch::indexing::Slice(startIdx, startIdx+batchLabels[i].sizes()[0])}, batchLabels[i]);
        predictions[i].index_put_({torch::indexing::Slice(startIdx, startIdx+taskLogits.sizes()[0])}, taskLogits);
        break;
			}

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
               std::vector<TaskWithCriterion> &tasks,
               std::vector<std::vector<float>> &losses,
               std::vector<torch::Tensor> &labels,
               std::vector<torch::Tensor> &predictions,
               torch::optim::Optimizer &optimizer) {
  model->train();
  classifier->train();
  auto callback = [&] (torch::Tensor loss) {
      model->zero_grad();
      loss.backward();
      optimizer.step();
  };
  innerLoop(model, classifier, loader, tasks, losses, labels, predictions, callback);
}

// Validation
void trainLoop(BertModel &model,
               BinaryClassifier &classifier,
               TextDataLoaderType &loader,
               std::vector<TaskWithCriterion> &tasks,
               std::vector<std::vector<float>> &losses,
               std::vector<torch::Tensor> &labels,
               std::vector<torch::Tensor> &predictions) {
  torch::NoGradGuard no_grad;
  model->eval();
  classifier->eval();
  auto callback = [] (torch::Tensor loss) {};
  innerLoop(model, classifier, loader, tasks, losses, labels, predictions, callback);
}
