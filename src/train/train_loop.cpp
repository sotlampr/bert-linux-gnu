#include <tuple>
#include <vector>
#include <stdexcept>
#include <torch/torch.h>
#include "data.h"
#include "model.h"


void innerLoop(BertModel &model,
               std::vector<Task> &tasks,
               TextDataLoaderType &loader,
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
					t = t.cuda();
			});

      torch::Tensor output = model->forward(data);
      torch::Tensor loss = torch::zeros(1, torch::requires_grad()).to(torch::kCUDA);
      torch::Tensor taskPredictions, taskLoss;
      
      for (size_t i = 0; i < tasks.size(); i++) {
        torch::Tensor taskLogits = tasks[i].classifier.forward(output);
        taskLoss = tasks[i].criterion.forward(taskLogits, batchLabels[i]);
        loss += taskLoss * tasks[i].lossMultiplier;
        losses[i].push_back(taskLoss.item<float>());
        labels[i].index_put_({torch::indexing::Slice(startIdx, startIdx+batchLabels[i].size(0))}, batchLabels[i]);

        taskPredictions = tasks[i].logitsToPredictions(taskLogits);
        predictions[i].index_put_({torch::indexing::Slice(startIdx, startIdx+taskPredictions.size(0))}, taskPredictions);
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
               std::vector<Task> &tasks,
               TextDataLoaderType &loader,
               std::vector<std::vector<float>> &losses,
               std::vector<torch::Tensor> &labels,
               std::vector<torch::Tensor> &predictions,
               torch::optim::Optimizer &optimizer) {
  model->train();
  for (auto& task : tasks) {
    task.classifier.ptr()->train();
  }
  auto callback = [&] (torch::Tensor loss) {
      model->zero_grad();
      for (auto& task : tasks) {
        task.classifier.ptr()->zero_grad();
      }
      loss.backward();
      optimizer.step();
  };
  innerLoop(model, tasks, loader, losses, labels, predictions, callback);
}

// Validation
void trainLoop(BertModel &model,
               std::vector<Task> &tasks,
               TextDataLoaderType &loader,
               std::vector<std::vector<float>> &losses,
               std::vector<torch::Tensor> &labels,
               std::vector<torch::Tensor> &predictions) {
  torch::NoGradGuard no_grad;
  model->eval();
  for (auto& task : tasks) {
    task.classifier.ptr()->eval();
  }
  auto callback = [] (torch::Tensor loss) {};
  innerLoop(model, tasks, loader, losses, labels, predictions, callback);
}
