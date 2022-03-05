"""
Just a small hacky script to make the train set smaller for Czech, for our experiments.
"""
import argparse
import os
import random
from typing import Dict, Tuple, List, Any, Callable

random.seed(1000)

def lazy_nonparse(text: str):
    for sentence in text.split("\n\n"):
        if sentence:
            yield [line for line in sentence.split("\n")]


annotations = []
with open(
    "../data/ud-treebanks-v2.3/UD_Czech-PDT/cs_pdt-ud-train.conllu", "r", encoding="utf-8"
) as conllu_file:
    for annotation in lazy_nonparse(conllu_file.read()):
        annotations.append(annotation)


random.shuffle(annotations)
development = annotations[:13000]

# new Dev
with open("../data/ud-treebanks-v2.3/UD_Czech-PDT/cs_pdt-ud-train_small.conllu", "w", encoding="utf-8") as f:
    for z in development:
        for line in z:
            f.write(line)
            f.write("\n")
        f.write("\n")  # break
