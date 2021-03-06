#include "train.h"

#include <getopt.h>
#include <iostream>

#include "data.h"
#include "task.h"
#include "train_utils.h"

#define CHECK_VECTOR_ARG(name, var) \
if (var.size() == 0) { \
  printHelp(argv[0]); \
  std::cout << "Error: `" << name "` required" << std::endl; \
  return 1; \
}

#define CHECK_STR_ARG(name, value) \
if (value == "") { \
  printHelp(argv[0]); \
  std::cout << "Error: `" << name "` required" << std::endl; \
  return 1; \
}

namespace train{

void printHelp(const std::string &programName) {
  std::cout << "Usage:" << std::endl;
  std::cout << programName <<" [OPTIONS]" << std::endl;
  std::cout << "\
Fine-tune a BERT model.\n\
See also `src/config.h` for compile-time options.\n\n\
Model and data selection:\n\
  -M, --model-dir           Extracted model directory\n\
                              The directory should include each model parameter in a\n\
                              `*.bin` 32-bit float binary file and the vocabulary in\n\
                              `vocab.txt`. Also see `extract_model.py`\n\
  -D, --data-dir            Base data directory\n\
                            The directory should include a `train-texts` and a \n\
                              `val-texts` file, as well as the equivalent\n\
                              `{train,val}-$task` files for each task\n\n\
Task selection:\n\
  -t  --task                Task name (see `--data-dir`)\n\
  -m  --metric              Add a metric for the specified task.\n\
                              Choose from: {accuracy,f1,matthewscc}\n\
  -l  --loss-multiplier     Multiplier for task loss\n\
                              Default: 0.1\n\n\
Training parameters:\n\
  -b, --batch-size\n\
  -e, --num-epochs\n\
  -a, --lr\n\n\
Training options:\n\
  -n, --num-workers         Number of workers for the data loader.\n\
                              Default: 0 (single-threaded)\n\
";
}

int main(int argc, char *argv[]) {
  int c, opt = 0;
  int batchSize = DEFAULT_BATCH_SIZE,
      numEpochs = DEFAULT_NUM_EPOCHS,
      numWorkers = 0, seed = 42;
  float lr = DEFAULT_LR;

  std::string modelDir, dataDir, saveModel; modelDir = dataDir = saveModel = "";
  std::vector<Task> tasks;
  Task lastTask;

	static struct option options[] = {
			{"batch-size",            required_argument, NULL,  'b' },
			{"num-epochs",            required_argument, NULL,  'e' },
			{"lr",                    required_argument, NULL,  'a' },
			{"num-workers",           required_argument, NULL,  'w' },
			{"model-dir",             required_argument, NULL,  'M' },
			{"save-model",            required_argument, NULL,  'S' },
			{"data-dir",              required_argument, NULL,  'D' },
			{"task",                  required_argument, NULL,  't' },
			{"metric",                required_argument, NULL,  'm' },
			{"loss-multiplier",       required_argument, NULL,  'l' },
			{"seed",                  required_argument, NULL,  's' },
			{"help",                  no_argument,       NULL,  'h' },
			{NULL,                    0,                 NULL,   0  }
	};

	while ((c = getopt_long(argc, argv, "-:b:e:a:w:M:S:D:t:m:l:s:h", options, &opt)) != -1) {
    switch (c) {
      case 1:
        printHelp(argv[0]);
        printf("Invalid argument '%s'\n", optarg); /* non-option arg */
        return 1;
     case 'b':
        batchSize = std::stoi(optarg);
        break;
      case 'e':
        numEpochs = std::stoi(optarg);
        break;
      case 'a':
        lr = std::stof(optarg);
        break;
      case 'w':
        numWorkers = std::stoi(optarg);
        break;
      case 'M':
        modelDir = optarg;
        break;
      case 'S':
        saveModel = optarg;
        break;
      case 'D':
        dataDir = optarg;
        lastTask.baseDir = dataDir;
        break;
      case 'h':
        printHelp(argv[0]);
        return 1;
      case 't':
        CHECK_STR_ARG("--data-dir", dataDir);
        if (!lastTask.name.empty()) {
          tasks.push_back(lastTask);
          lastTask = Task();
          lastTask.baseDir = dataDir;
        }
        lastTask.name = optarg;
        break;
      case 'm':
        if (lastTask.name.empty()) {
          printf("Task not specified, cannot add metric `%s`\n", optarg);
          return 1;
        }
        lastTask.addMetric(optarg);
        break;
      case 'l':
        if (lastTask.name.empty()) {
          printf("Task not specified, cannot set loss multiplier `%s`", optarg);
          return 1;
        }
        lastTask.lossMultiplier = std::stof(optarg);
        break;
      case 's':
        seed = std::stoi(optarg);
        break;
      case '?':
        printHelp(argv[0]);
        printf("Invalid argument -%c\n", optopt);
        return 1;
      case ':':
        printHelp(argv[0]);
        printf("Missing option for %c\n", optopt);
        return 1;
      default:
        printf("?? getopt returned character code 0%o ??\n", c);
        return 1;
		 }
  }

  if (!lastTask.name.empty()) {
    tasks.push_back(lastTask);
  }

  if (tasks.size() == 1) {
    tasks[0].lossMultiplier = 1.0f;
  }


  CHECK_STR_ARG("--model-dir", modelDir);
  CHECK_STR_ARG("--data-dir", dataDir);
  CHECK_VECTOR_ARG("--task", tasks);

  for (auto& task : tasks) {
    detectTaskType(task);
    std::cerr << "# "
              << "task=" << task.name
              << " loss_multiplier=" << task.lossMultiplier
              << " base_dir=" << task.baseDir
              << " seed=" << seed;
    for (const auto& metric : task.metrics) {
      std::cerr << " metric=" << metric.first;
    }
    std::cerr << " regression?=" << ((Regression & task.taskType) == Regression)
              << " token_level?=" << ((TokenLevel & task.taskType) == TokenLevel)
              << " multi_label?=" << ((MultiLabel & task.taskType) == MultiLabel)
              << " binary?=" << ((Binary & task.taskType) == Binary)
              << std::endl;
  }

  runTraining(modelDir, dataDir, tasks, batchSize, numEpochs, lr, numWorkers,
              saveModel, seed);

 return 0;
}
}
