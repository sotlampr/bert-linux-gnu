#!/usr/bin/env python3
import argparse
import re
import sys

import spacy


def get_pos(token, iob):
    return iob + "-" + token.pos_


def get_dep(token, iob):
    return iob + "-" + token.dep_


def get_ent(token, iob):
    if not token.ent_type_:
        return "O"
    iob = "I" if iob == "I" and token.ent_type_ else token.ent_iob_
    return iob + "-" + token.ent_type_


def main(args):
    nlp = spacy.load(args.spacy_model)
    if args.pos:
        get_func = get_pos
    elif args.ner:
        get_func = get_ent
    elif args.deps:
        get_func = get_dep
    for line in sys.stdin:
        tokenized_text = line.strip()
        n_tokens = len(tokenized_text.split())
        full_text = re.sub(r" ##([^\s])", r"\1", tokenized_text)
        doc = nlp(full_text)
        tags = []
        for i, token in enumerate(doc):
            if token.text == "SEP":
                # [SEP] instance (previous token was '[')
                # Change last tag
                tags[-1] = "SPECIAL_TOKEN"
            elif i == 0:
                # Beginning of sentence, pos tag is valid
                tags.append(get_func(token, "B"))
            elif tokenized_text[0] == " " and tokenized_text[1:3] != "##":
                # End of word, pos tag is valud
                tags.append(get_func(token, "B"))
                tokenized_text = tokenized_text[1:]

            for char in token.text:
                if char == tokenized_text[0]:
                    # char is the first from tokenized_text, move on
                    tokenized_text = tokenized_text[1:]
                elif tokenized_text[:3] == " ##":
                    # tokenized_text starts with ` ##`, append I(nside) tag
                    assert(char == tokenized_text[3])
                    tokenized_text = tokenized_text[4:]
                    tags.append(get_func(token, "I"))
        assert(len(tags) == n_tokens)
        print(args.delimiter.join(tags))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract POS tags using SpaCy."
                    "Reads stdin, writes to stdout.")
    parser.add_argument("spacy_model")
    parser.add_argument("-d", "--delimiter", default=",")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-P', "--pos", action='store_true')
    group.add_argument('-N', "--ner", action='store_true')
    group.add_argument('-D', "--deps", action='store_true')
    main(parser.parse_args())
