#!/usr/bin/env python3
import argparse
import re
import sys

import spacy


def main(args):
    nlp = spacy.load(args.spacy_model, disable=["parser", "ner", "textcat"])
    for line in sys.stdin:
        tokenized_text = line.strip()
        full_text = re.sub(r" ##([^\s])", r"\1", tokenized_text)
        tokenized_text = re.sub(r" ", "", tokenized_text)
        doc = nlp(full_text)
        tags = []
        for token in doc:
            pos_tag = token.pos_
            tags.append(f"B-{pos_tag}")
            for char in token.text:
                if char == tokenized_text[0]:
                    tokenized_text = tokenized_text[1:]
                else:
                    # Skip `##`
                    tokenized_text = tokenized_text[3:]
                    tags.append(f"I-{pos_tag}")
        print(args.delimiter.join(tags))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract POS tags using SpaCy."
                    "Reads stdin, writes to stdout.")
    parser.add_argument("spacy_model")
    parser.add_argument("-d", "--delimiter", default=",")
    main(parser.parse_args())
