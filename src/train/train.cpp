#include <getopt.h>
#include <iostream>
#include "config.h"
#include "train_utils.h"

#define CHECK_INT_ARG(name, value) \
if (value == -1) { \
  std::cout << "Error: `" << name "` required" << std::endl; \
  return 1; \
}

#define CHECK_STR_ARG(name, value) \
if (value == "") { \
  std::cout << "Error: `" << name "` required" << std::endl; \
  return 1; \
}

void printHelp(const std::string &programName) {
  std::cout << "Usage:" << std::endl;
  std::cout << programName <<" [OPTIONS]" << std::endl;
  std::cout << "\
Fine-tune a BERT model.\n\
See also `src/config.h` for compile-time options.\n\n\
Model and task selection:\n\
  -M, --model-dir   Extracted model directory\n\
                    The directory should include each model parameter in a\n\
                    `*.bin` 32-bit float binary file and the vocabulary in\n\
                    `vocab.txt`. Also see `extract_model.py`\n\
  -D, --data-dir    Base data directory\n\
                    The directory should include a `train-texts` and a \n\
                    `val-texts` file, as well as the equivalent\n\
                    `{train,val}-$task` files for each task\n\n\
Training parameters:\n\
  -b, --batch-size\n\
  -e, --num-epochs\n\n\
Training options:\n\
  -n, --num-workers Number of workers for the data loader.\n\
                      Default: 0 (single-threaded)\n\
";
}

int main(int argc, char *argv[]) {
  int c, opt = 0;
  int batchSize, numEpochs; batchSize = numEpochs = -1;
  int numWorkers = 0;
  std::string modelDir, dataDir; modelDir = dataDir = "";

	static struct option options[] = {
			{"batch-size",   required_argument, NULL,  'b' },
			{"num-workers",  required_argument, NULL,  'w' },
			{"num-epochs",   required_argument, NULL,  'e' },
			{"model-dir",    required_argument, NULL,  'M' },
			{"data-dir",     required_argument, NULL,  'D' },
			{"help",         no_argument,       NULL,  'h' },
			{NULL,          0,                 NULL,   0  }
	};
	while ((c = getopt_long(argc, argv, "-:b:e:w:M:D:h", options, &opt)) != -1) {
    switch (c) {
      case 1:
        printf("Invalid argument '%s'\n", optarg); /* non-option arg */
        return 1;
     case 'b':
        batchSize = std::stoi(optarg);
        break;
      case 'e':
        numEpochs = std::stoi(optarg);
        break;
      case 'w':
        numWorkers = std::stoi(optarg);
        break;
      case 'M':
        modelDir = optarg;
        break;
      case 'D':
        dataDir = optarg;
        break;
      case 'h':
        printHelp(argv[0]);
        break;
      case '?':
        printf("Invalid argument -%c\n", optopt);
        return 1;
      case ':':
        printf("Missing option for %c\n", optopt);
        return 1;
      default:
        printf("?? getopt returned character code 0%o ??\n", c);
        return 1;
		 }
  }

  CHECK_INT_ARG("--batch-size", batchSize);
  CHECK_INT_ARG("--num-epochs", numEpochs);
  CHECK_STR_ARG("--model-dir", modelDir);
  CHECK_STR_ARG("--data-dir", dataDir);

  Config config;
  runTraining(config, modelDir, dataDir, batchSize, numWorkers, numEpochs);

  return 0;
}
