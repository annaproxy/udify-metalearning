"""
Concatenates all treebanks together
"""
from __future__ import unicode_literals
import os
import shutil
import logging
import argparse

from udify import util

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output_dir", type=str, help="The path to output the concatenated files"
)
parser.add_argument(
    "--dataset_dir",
    default="data/expmix",
    type=str,
    help="The path containing all UD treebanks",
)
parser.add_argument(
    "--treebanks",
    default=[],
    type=str,
    nargs="+",
    help="Specify a list of treebanks to use; leave blank to default to all treebanks available",
)

args = parser.parse_args()

treebanks = util.get_ud_treebank_files(args.dataset_dir, args.treebanks)
train, dev, test = list(zip(*[treebanks[k] for k in treebanks]))

for treebank, name in zip(
    [train, dev, test], ["train.conllu", "dev.conllu", "test.conllu"]
):
    with open(os.path.join(args.output_dir, name), "w", encoding="utf-8") as write:
        for t in treebank:
            if not t:
                continue
            with open(t, "r", encoding="utf-8") as read:
                # Unicode hell on Lisa, if this throws an error then just remove it,
                # but watch out with unicode !!! 
                shutil.copyfileobj(unicode(read), write)
