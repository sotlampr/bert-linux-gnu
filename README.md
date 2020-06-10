BERT \& libtorch

# Motivation

Inspired by the [UNIX philosophy](https://en.wikipedia.org/wiki/Unix_philosophy):

- Write programs that do one thing well.
- Write programs that work well together
- Write programs to handle text streams, because that is a universal interface.

The one thing that `bert-linux-gnu` does well is multi-task learning using BERT
and [libtorch](https://pytorch.org/cppdocs/) on a single GPU:

- ✓ Simple command-line interface
- ✓ Use [pytorch-transformers](https://github.com/huggingface/transformers) models out-the-box
- ✓ Run new tasks and datasets without writing a single line of code
- ✓ Out-of-the-box multi-task learning support
- ✓ Out-of-the-box task type detection based on file structure
- ✓ Basic options via the command line, advanced by editing `config.h` and recompiling
- ✗ Multi-GPU
- ✗ Only original BERT
- ✗ Fancy optimizers
- ✗ Fancy output
- ✗ Tensorflow, Keras, Scikit-learn, you name it

# Prerequisites

- ICU
- pytorch
- pytorch-transformers
- Recent gcc, stdlibc++ etc.

# Example

- Compile:

`make -j$(nproc) all`

- Download and preprocess GLUE data:

`make glue`

- Extract a model from `pytorch-transformers`:

`./python_utils/extract_model.py bert-base-uncased`

- Run CoLA:

```
$ ./bert train \
  --batch-size=32 \
  --num-epochs=10 \
  --model-dir=models/bert-base-uncased \
  --data-dir=glue/data/CoLa/processed \
  --task acceptability \
  --metric mathewscc | cut -f5 -d,'

> # task=acceptability loss_multiplier=1 base_dir=glue/data/CoLA/processed/ metric=matthewscc regression?=0 token_level?=0 binary?=1
> acceptability_val_matthewscc
> 0.502576
> 0.561762
> 0.586907
> 0.603412
> 0.549615
> 0.584714
> 0.573691
> 0.602503
> 0.573091
> 0.568567
```

- Run MRPC:

```
$ ./bert train \
  --batch-size=32 \
  --num-epochs=10 \
  --model-dir=models/bert-base-uncased \
  --data-dir=glue/data/MRPC/processed \
  --task paraphrase \
  --metric accuracy \
  --metric f1 | cut -f7 -d','

> # task=paraphrase loss_multiplier=1 base_dir=glue/data/MRPC/processed/ metric=accuracy metric=f1 regression?=0 token_level?=0 binary?=1
> WARNING: truncating sequence to 100
> paraphrase_val_f1
> 0.772549
> 0.875657
> 0.87037
> 0.889292
> 0.890459
> 0.892308
> 0.887719
> 0.892473
> 0.886076
> 0.893238
```

# Implemented

- BERT tokenizer
- BERT model
- Binary \& Multi-class sentence-level classification
- Multi-sentence tasks (via manual preprocessing)

# Will implement

- SentencePiece tokenizer (for models trained with SP-tokenized texts)
- Token-level classification
- Save model
- Test model
- Multi-class metrics

# Will not implement

- Multi-GPU
- Preprocessing routines (use *NIX tools!)
- BERT pre-training
