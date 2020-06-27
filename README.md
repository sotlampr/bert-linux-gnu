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
- ✓ .csv-ready output
- ✓ Basic options via the command line, advanced by editing `config.h` and recompiling
- ✗ Multi-GPU
- ✗ Only original BERT
- ✗ Fancy optimizers
- ✗ Fancy output
- ✗ Tensorflow, Keras, Scikit-learn, you name it

# Prerequisites

- [libtorch (w/ CXX ABI)](https://pytorch.org/get-started/locally/)
- [sentencepiece](https://github.com/google/sentencepiece)
- pytorch & pytorch-transformers
- ICU
- Recent gcc, stdlibc++ etc.

# Examples

- Compile:

`$ make -j$(nproc) all`

- Download and preprocess GLUE data:

`$ make glue`

- Extract a model from `pytorch-transformers`:

`$ ./python_utils/extract_model.py bert-base-uncased`

- Run CoLA:

```
$ ./bert train \
  --batch-size=32 \
  --num-epochs=10 \
  --model-dir=models/bert-base-uncased \
  --data-dir=glue/data/CoLa/processed \
  --task acceptability \
  --metric mathewscc | cut -f1,5 -d,'

> WARNING: Parameter pooler.dense.bias not in model
> WARNING: Parameter pooler.dense.weight not in model
> # task=acceptability loss_multiplier=1 base_dir=glue/data/CoLA/processed/ metric=matthewscc regression?=0 token_level?=0 binary?=1
> epoch,acceptability_val_matthewscc
> 1,0.5595
> 2,0.574558
> 3,0.589253
> 4,0.585463
> 5,0.583288
> 6,0.54521
> 7,0.581384
> 8,0.586692
> 9,0.570512
> 10,0.571522
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
  --metric f1 | cut -f1,6,7 -d','

> # task=paraphrase loss_multiplier=1 base_dir=glue/data/MRPC/processed/ metric=accuracy metric=f1 regression?=0 token_level?=0 binary?=1
> WARNING: Parameter pooler.dense.bias not in model
> WARNING: Parameter pooler.dense.weight not in model
> WARNING: truncating sequence to 100
> epoch,paraphrase_val_accuracy,paraphrase_val_f1
> 1,0.715686,0.810458
> 2,0.811275,0.86747
> 3,0.82598,0.869245
> 4,0.830882,0.872458
> 5,0.848039,0.891228
> 6,0.833333,0.885906
> 7,0.828431,0.879725
> 8,0.843137,0.883212
> 9,0.828431,0.880546
> 10,0.828431,0.881757
```

## Multi-task fun - add some syntax to the mix!

- Use spacy to extract POS tags for the CoLA dataset:

```
$ for split in train val; do
  ./bert tokenize \
    models/bert-base-uncased \
    glue/data/CoLA/processed/${split}-texts |
    parallel --pipe -j$(nproc) \
      ./python_utils/extract_pos.py en_core_web_lg > cola-${split}-postags
done
```

- Convert the POS tags to ids:

```
$ ./python_utils/convert_to_ids.py \
  cola-train-postags \
  glue/data/CoLA/processed/train-pos \
  -s cola-postags.vocab

$ ./python_utils/convert_to_ids.py \
  cola-val-postags \
  glue/data/CoLA/processed/val-pos \
  -v cola-postags.vocab
```

- Run CoLA acceptability + POS tagging

```
$ ./bert train \
  --batch-size=32 \
  --num-epochs=10 \
  --model-dir=models/bert-base-uncased \
  --data-dir=glue/data/CoLa/processed \
  --task acceptability \
  --loss-multiplier 1.0 \
  --metric mathewscc \
  --task pos \
  --loss-multiplier 0.1 \
  --metric accuracy | cut -f1,7,9 -d','

> # task=acceptability loss_multiplier=1 base_dir=glue/data/CoLA/processed/ metric=matthewscc regression?=0 token_level?=0 binary?=1
> # task=pos loss_multiplier=0.1 base_dir=glue/data/CoLA/processed/ metric=accuracy regression?=0 token_level?=1 binary?=0
> WARNING: Parameter pooler.dense.bias not in model
> WARNING: Parameter pooler.dense.weight not in model
> epoch,acceptability_val_matthewscc,pos_val_accuracy
> 1,0.504317,0.634615
> 2,0.579998,0.77617
> 3,0.581538,0.853192
> 4,0.567737,0.885012
> 5,0.546342,0.900773
> 6,0.588415,0.908109
> 7,0.533828,0.913561
> 8,0.552374,0.924366
> 9,0.566321,0.926546
> 10,0.554061,0.930511
```

- Similarly for MRPC:

```
$ ./bert train \
  --batch-size=32 \
  --num-epochs=10 \
  --model-dir=models/bert-base-uncased \
  --data-dir=glue/data/MRPC/processed \
  --task paraphrase \
  --loss-multiplier 1.0 \
  --metric accuracy \
  --metric f1 \
  --task pos \
  --loss-multiplier 0.1 \
  --metric accuracy | cut -f1,8,9,11 -d','

> # task=paraphrase loss_multiplier=1 base_dir=glue/data/MRPC/processed/ metric=accuracy metric=f1 regression?=0 token_level?=0 binary?=1
> # task=pos loss_multiplier=0.1 base_dir=glue/data/MRPC/processed/ metric=accuracy regression?=0 token_level?=1 binary?=0
> WARNING: Parameter pooler.dense.bias not in model
> WARNING: Parameter pooler.dense.weight not in model
> WARNING: truncating sequence to 100
> WARNING: truncating sequence to 100
> epoch,paraphrase_val_accuracy,paraphrase_val_f1,pos_val_accuracy
> 1,0.757353,0.820327,0.300746
> 2,0.796569,0.8431,0.430021
> 3,0.85049,0.89317,0.528562
> 4,0.791667,0.832347,0.605981
> 5,0.848039,0.892734,0.681838
> 6,0.845588,0.890815,0.740817
> 7,0.857843,0.899654,0.775745
> 8,0.860294,0.90087,0.806332
> 9,0.857843,0.9,0.826382
> 10,0.85049,0.896082,0.842139
```

Whoa, that was nice!!

# Implemented

- BERT tokenizer
- BERT model
- Binary \& Multi-class sentence-level classification
- Multi-sentence tasks (via manual preprocessing)
- Token-level calssification
- `AdamW` optimizer
- Gradient clipping

# Will implement

- SentencePiece tokenizer (for models trained with SP-tokenized texts)
- Test model
- Multi-class metrics

# Will not implement

- Multi-GPU
- Preprocessing routines (use *NIX tools!)
- BERT pre-training
