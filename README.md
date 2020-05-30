pytorch-transformers \& libtorch

# Prerequisites

- ICU

# Implemented

## Tokenize

```
time python -c "\
from bert.tokenization import FullTokenizer
tokenizer = FullTokenizer('uncased_L-12_H-768_A-12/vocab.txt', True)
with open('~/a-file.txt') as fp:
  for line in fp.readlines():
    tokenizer.tokenize(line)
"  
> real    2m47.340s
> user    2m47.981s
> sys     0m1.472s
```


```
make tokenize && \
time ./tokenize a-file.txt >/dev/null

> real    0m38.205s
> user    0m38.085s
> sys     0m0.120s
```
