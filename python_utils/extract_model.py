#!/usr/bin/env python3
import argparse
import array
import os
import re
import shutil

import torch
from pytorch_transformers import BertModel, BertTokenizer


def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    vocab_file = tokenizer \
        .pretrained_init_configuration["bert-base-uncased"]["vocab_file"]
    model_dir = args.dir + "/" + args.model_name
    os.makedirs(model_dir, exist_ok=True)
    shutil.copy2(vocab_file, f"{model_dir}/vocab.txt")

    model = BertModel.from_pretrained(args.model_name)
    for k, v in model.named_parameters():
        if isinstance(v, torch.Tensor):
            values = array.array("f", v.detach().numpy().ravel())
            k = "".join(map(lambda x: x[0].upper()+x[1:], k.split("_")))
            k = k[0].lower() + k[1:]
            k = re.sub("LayerNorm", "layerNorm", k)
            fname = f"{model_dir}/{k}-{'_'.join(map(str, v.size()))}.dat"
            with open(fname, "wb") as fp:
                values.tofile(fp)
        else:
            print(f"error for {k}")
    print(f"Extracted {args.model_name} to {model_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Extract a pytorch-transformers BERT model's weight and vocabulary")
    parser.add_argument("model-name")
    parser.add_argument("-d", "--dir", default="./models")
    main(parser.parse_args())
