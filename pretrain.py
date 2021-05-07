"""
Simply pre-training using UDifys regime
"""

import os
import copy
import datetime
import logging
import argparse
import sys

from allennlp.common import Params
from allennlp.models.model import Model
from allennlp.common.util import import_submodules
from allennlp.commands.train import train_model

from udify import util

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name", default="hindi_pretrained", type=str, help="Log dir name"
)
parser.add_argument(
    "--base_config",
    default="config/udify_base.json",
    type=str,
    help="Base configuration file",
)
parser.add_argument(
    "--config",
    default="config/ud/en/udify_bert_finetune_hindi.json",
    type=str,
    nargs="+",
    help="Overriding configuration files",
)
parser.add_argument(
    "--device", default=None, type=int, help="CUDA device; set to -1 for CPU"
)
parser.add_argument("--resume", type=str, help="Resume training with the given model")
parser.add_argument(
    "--lazy", default=None, action="store_true", help="Lazy load the dataset"
)
parser.add_argument(
    "--cleanup_archive", action="store_true", help="Delete the model archive"
)
# parser.add_argument("--replace_vocab", action="store_true", help="Create a new vocab and replace the cached one")
parser.add_argument(
    "--archive_bert",
    action="store_true",
    help="Archives the finetuned BERT model after training",
)
parser.add_argument(
    "--predictor",
    default="udify_predictor",
    type=str,
    help="The type of predictor to use",
)
parser.add_argument(
    "--seed", default=1, type=int, help="Seed to use for pytorch,python and numpy"
)
args = parser.parse_args()

log_dir_name = args.name
if not log_dir_name:
    file_name = args.config[0] if args.config else args.base_config
    log_dir_name = os.path.basename(file_name).split(".")[0]

configs = []

if not args.resume:
    serialization_dir = os.path.join(
        "logs", log_dir_name, datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    )

    overrides = {
        "random_seed": args.seed,
        "numpy_seed": args.seed,
        "pytorch_seed": args.seed,
    }
    if args.device is not None:
        overrides["trainer"] = {"cuda_device": args.device}
    if args.lazy is not None:
        overrides["dataset_reader"] = {"lazy": args.lazy}
    configs.append(Params(overrides))
    # configs.append(Params({'config':}))
    configs.append(Params.from_file(args.config))
    configs.append(Params.from_file(args.base_config))
else:
    serialization_dir = args.resume
    configs.append(Params.from_file(os.path.join(serialization_dir, "config.json")))

train_params = util.merge_configs(configs)
predict_params = train_params.duplicate()
print("Starting with seed", train_params["random_seed"])

if "vocabulary" in train_params:
    # Remove this key to make AllenNLP happy
    train_params["vocabulary"].pop("non_padded_namespaces", None)

import_submodules("udify")


try:
    # Vocabulary should be there already! Built by us on the specific subset!
    # util.cache_vocab(train_params)
    train_model(train_params, serialization_dir, recover=bool(args.resume))
except KeyboardInterrupt:
    logger.warning("KeyboardInterrupt, skipping training")

if args.archive_bert:
    bert_config = "config/archive/bert-base-multilingual-cased/bert_config.json"
    util.archive_bert_model(serialization_dir, bert_config)

util.cleanup_training(serialization_dir, keep_archive=not args.cleanup_archive)
