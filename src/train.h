#include <

int main() {
  Config config;
  BertModel model(config);
  loadState("models/bert-base-uncased", *model);
  return 0;
}
