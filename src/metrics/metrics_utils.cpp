#include "metrics_utils.h"

torch::Tensor logicalAnd(torch::Tensor &a, torch::Tensor &b) {
  torch::Tensor zeros = torch::zeros_like(a);
  torch::Tensor ones = torch::ones_like(a);
  torch::Tensor out = torch::where(a, ones, zeros);
  out = torch::where(b, out, zeros);
  return out;

}

long truePositives(torch::Tensor &labels, torch::Tensor &predictions) {
  torch::Tensor a = (labels == 1).to(torch::kByte);
  torch::Tensor b = (predictions == 1).to(torch::kByte);
  return logicalAnd(a, b).sum().item<long>();
}

long falsePositives(torch::Tensor &labels, torch::Tensor &predictions) {
  torch::Tensor a = (labels == 0).to(torch::kByte);
  torch::Tensor b = (predictions == 1).to(torch::kByte);
  return logicalAnd(a, b).sum().item<long>();
}

long trueNegatives(torch::Tensor &labels, torch::Tensor &predictions) {
  torch::Tensor a = (labels == 0).to(torch::kByte);
  torch::Tensor b = (predictions == 0).to(torch::kByte);
  return logicalAnd(a, b).sum().item<long>();
}

long falseNegatives(torch::Tensor &labels, torch::Tensor &predictions) {
  torch::Tensor a = (labels == 1).to(torch::kByte);
  torch::Tensor b = (predictions == 0).to(torch::kByte);
  return logicalAnd(a, b).sum().item<long>();
}
