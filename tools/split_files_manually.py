"""
Split test files for language that only have a test set, such as Swedish, Faroese and Breton.
"""
import argparse
import os
import random
from typing import Dict, Tuple, List, Any, Callable


def lazy_nonparse(text: str):
    for sentence in text.split("\n\n"):
        if sentence:
            yield [line for line in sentence.split("\n")]


parser = argparse.ArgumentParser()
parser.add_argument("--file", default=None, type=str, help="File to split up")
parser.add_argument("--amt", default=20, type=int, help="How many pieces to split up")
parser.add_argument("--batch_size", default=20, type=int, help="How big is each batch")
parser.add_argument("--seed", default=2002, type=int, help="Seed for the shuffler")

args = parser.parse_args()
output_filename = os.path.join(
    "../data/ud-tiny-treebanks/size" + str(args.batch_size),
    args.file.strip("-test.conllu").split("/")[-1],
)

annotations = []
with open(args.file, "r", encoding="utf-8") as conllu_file:
    for annotation in lazy_nonparse(conllu_file.read()):
        annotations.append(annotation)


for i in range(args.amt):
    random.shuffle(annotations)
    development = annotations[: args.batch_size]
    test = annotations[args.batch_size :]

    # new Dev
    with open(output_filename + "-dev" + str(i) + ".conllu", "w") as f:
        for z in development:
            for line in z:
                f.write(line)
                f.write("\n")
            f.write("\n")
    # new Test
    with open(output_filename + "-test" + str(i) + ".conllu", "w") as f:
        for z in test:
            for line in z:
                f.write(line)
                f.write("\n")
            f.write("\n")
