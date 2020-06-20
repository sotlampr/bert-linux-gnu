#include "train_loop.h"

#include <tuple>
#include <vector>
#include <stdexcept>

#include <torch/nn/utils.h>

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

      // Move the labels to CUDA
			std::for_each(batchLabels.begin(), batchLabels.end(), 
				[](torch::Tensor &t) {
					t = t.cuda();
			});

      torch::Tensor output = model->forward(data);

      // Total loss placeholder
      torch::Tensor loss = torch::zeros(1, torch::requires_grad()).cuda();

      torch::Tensor taskLogits, taskLoss, taskPredictions;
      for (size_t i = 0; i < tasks.size(); i++) {
        taskLogits = tasks[i].classifier.forward(output);
        if (taskLogits.ndimension() == 3) {
          // Token-level, flatten logits and targets
          taskLoss = tasks[i].criterion.forward(
            taskLogits.view({-1, taskLogits.size(2)}),
            batchLabels[i].view({-1})
          );
        } else {
          taskLoss = tasks[i].criterion.forward(taskLogits, batchLabels[i]);
        }
        loss += taskLoss * tasks[i].lossMultiplier;
        losses[i].push_back(taskLoss.item<float>());
        
        // Insert the true labels for the batch to `labels`
        labels[i].index_put_({torch::indexing::Slice(startIdx, startIdx+batchLabels[i].size(0))}, batchLabels[i]);

        // Convert the task logits to predicted classes
        taskPredictions = tasks[i].logitsToPredictions(taskLogits);

        // Insert the predicted labels for the batch to `predictions`
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

  // Set all classifier heads to train mode
  for (auto& task : tasks) {
    task.classifier.ptr()->train();
  }
  
  // Training callback - perform an optimization step
  auto callback = [&] (torch::Tensor loss) {
      model->zero_grad();
      for (auto& task : tasks) {
        task.classifier.ptr()->zero_grad();
      }
      loss.backward();
      // Gradient clipping
      for (auto& param_group : optimizer.param_groups()) {
        for (auto& param : param_group.params()) {
          torch::nn::utils::clip_grad_norm_(param, MAX_GRADIENT_NORM);
        }
      }
      optimizer.step();
  };

  // Train for an epoch
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
  // Set all classifier heads to eval mode
  for (auto& task : tasks) {
    task.classifier.ptr()->eval();
  }
  auto callback = [] (torch::Tensor loss) {}; // Dummy callback, does nothing

  // Forward for an epoch
  innerLoop(model, tasks, loader, losses, labels, predictions, callback);
}
