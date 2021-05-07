import os
import datetime
import argparse
from allennlp.common import Params
from udify import util


def get_params(name, seed):
    """Some default parameters.
    Note that this will initially include training parameters that you won't need for metalearning since we have our own training loop."""
    configs = []
    overrides = {}

    overrides["dataset_reader"] = {"lazy": True}

    configs.append(Params(overrides))
    configs.append(
        Params({"random_seed": seed, "numpy_seed": seed, "pytorch_seed": seed})
    )
    configs.append(Params.from_file("config/ud/en/udify_bert_finetune_en_ewt.json"))
    configs.append(Params.from_file("config/udify_base.json"))

    return util.merge_configs(configs)
