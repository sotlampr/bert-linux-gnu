#!/usr/bin/env python3
import argparse
import array
import os
import re
import shutil
import struct

import torch
from pytorch_transformers import BertConfig, BertModel, BertTokenizer
from pytorch_transformers.tokenization_bert \
    import PRETRAINED_INIT_CONFIGURATION


def save_struct(config, model_dir):
    with open(f"{model_dir}/config", "wb") as fp:
        fp.write(
            struct.pack("@iffiiiiii",
                config.hidden_size,
                config.attention_probs_dropout_prob,
                config.hidden_dropout_prob,
                config.intermediate_size,
                config.max_position_embeddings,
                config.num_attention_heads,
                config.num_hidden_layers,
                config.type_vocab_size,
                config.vocab_size
            )
        )


def main(args):
    args.model_name = args.model_name if args.model_name else args.model
    model_dir = args.dir + "/" + args.model_name
    os.makedirs(model_dir, exist_ok=True)

    if args.sentencepiece:
        shutil.copy2(args.sentencepiece, f"{model_dir}/model.sp")
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model)
        vocab_file = tokenizer \
            .pretrained_init_configuration["bert-base-uncased"]["vocab_file"]
        os.makedirs(model_dir, exist_ok=True)
        shutil.copy2(vocab_file, f"{model_dir}/vocab.txt")

    if PRETRAINED_INIT_CONFIGURATION[args.model]["do_lower_case"]:
        open(f"{model_dir}/lowercase", 'w').close()

    config = BertConfig.from_pretrained(args.model)
    if args.checkpoint:
        state_dict = {k.replace("bert.", ""): v
                      for (k, v) in torch.load(args.checkpoint).items()
                      if "bert." in k}
        config.vocab_size = (
            state_dict["embeddings.word_embeddings.weight"].size(0))
        model = BertModel(config)
        model.load_state_dict(state_dict)
    else:
        model = BertModel.from_pretrained(args.model)

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

    save_struct(config, model_dir)
    print(f"Extracted `{args.model_name}` to `{model_dir}`")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Extract a pytorch-transformers BERT model's weight and vocabulary")
    parser.add_argument("model")
    parser.add_argument("-m", "--model_name")
    parser.add_argument("-c", "--checkpoint")
    parser.add_argument("-s", "--sentencepiece")
    parser.add_argument("-d", "--dir", default="./models")
    main(parser.parse_args())
