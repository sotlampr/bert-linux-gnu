#!/usr/bin/env python3
import argparse
import csv
import shutil


def main(args):
    with open(args.fname, newline="") as fp:
        reader = csv.reader(fp, delimiter=args.delimiter)
        data = list(reader)

    if args.vocabulary_fname:
        with open(args.vocabulary_fname) as fp:
            vocabulary = {k: v
                          for (k, v) in map(lambda x: x.split(","),
                                            fp.read().strip().split("\n"))}
    else:
        values = set([y for x in data for y in x])
        vocabulary = {x: i for i, x in enumerate(sorted(values))}

    data = [[vocabulary.get(y, args.unk_id) for y in x] for x in data]

    if args.backup:
        shutil.copyfile(args.fname, args.fname + ".bak")

    if args.save_vocabulary_fname:
        with open(args.save_vocabulary_fname, "w") as fp:
            for k, v in vocabulary.items():
                print(f"{k},{v}", file=fp)

    with open(args.fname, "w", newline="") as fp:
        writer = csv.writer(fp, delimiter=args.delimiter)
        for row in data:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a file containing labels to integer ids. "
                    "Overwrites the original file.")
    parser.add_argument("fname")
    parser.add_argument("-v", "--vocabulary-fname")
    parser.add_argument("-s", "--save-vocabulary-fname")
    parser.add_argument("-b", "--backup", action="store_true")
    parser.add_argument("-d", "--delimiter", default=",")
    parser.add_argument("-u", "--unk-id", type=int, default=-1)
    main(parser.parse_args())
