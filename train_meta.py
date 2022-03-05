# -*- coding: utf-8 -*-
"""
This file Meta-Trains on 7 languages
And validates on Bulgarian
"""
from naming_conventions import train_languages, train_languages_lowercase
from get_language_dataset import get_language_dataset
from get_default_params import get_params
from udify import util
from ourmaml import MAML
from udify.predictors import predictor
from allennlp.common.util import prepare_environment
from allennlp.models.model import Model
from allennlp.models.archival import archive_model
import allennlp
from schedulers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from torch import autograd
from torch.optim import Adam
import torch
import numpy as np
import argparse
import subprocess
import json
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip_update", default=0, type=float, help="Skip update on the support set"
    )
    parser.add_argument("--seed", default=9999, type=int, help="Set seed")
    parser.add_argument(
        "--skip_update", default=0, type=float, help="Skip update on the support set"
    )

    parser.add_argument(
        "--support_set_size", default=32, type=int, help="Support set size"
    )
    parser.add_argument(
        "--maml",
        default=False,
        type=bool,
        help="Do MAML instead of XMAML, that is, include English as an auxiliary task if flag is set and start from scratch",
    )
    parser.add_argument("--addenglish", default=False,
                        type=bool, help="Add English as a task")
    parser.add_argument("--notaddhindi", default=False,
                        type=bool, help="Skip Hindi as a task")

    parser.add_argument("--notadditalian", default=False,
                        type=bool, help="Skip Italian as a task")
    parser.add_argument("--notaddczech", default=False,
                        type=bool, help="Skip Czech as a task")

    parser.add_argument("--episodes", default=900,
                        type=int, help="Amount of episodes")
    parser.add_argument(
        "--updates", default=5, type=int, help="Amount of inner loop updates"
    )
    parser.add_argument("--name", default="", type=str, help="Name to add")
    parser.add_argument(
        "--meta_lr_decoder",
        default=None,
        type=float,
        help="Meta adaptation LR for the decoder",
    )
    parser.add_argument(
        "--meta_lr_bert", default=None, type=float, help="Meta adaptation LR for BERT"
    )
    parser.add_argument(
        "--inner_lr_decoder",
        default=None,
        type=float,
        help="Inner learner LR for the decoder",
    )
    parser.add_argument(
        "--inner_lr_bert", default=None, type=float, help="Inner learner LR for BERT"
    )

    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        help="Directory from where to start training. Should be a 'clean' model for MAML and a pretrained model for X-MAML.",
    )
    args = parser.parse_args()

    # Yes, i know.
    training_tasks = []
    train_languages = np.array(train_languages)
    train_languages_lowercase = np.array(train_languages_lowercase)
    hindi_indices = [0, 1, 2, 3, 4, 6]
    italian_indices = [0, 1, 3, 4, 5, 6]
    czech_indices = [0, 2, 3, 4, 5, 6]
    if args.notaddhindi:
        train_languages = train_languages[hindi_indices]
        train_languages_lowercase = train_languages_lowercase[hindi_indices]
    elif args.notaddczech:
        train_languages = train_languages[czech_indices]
        train_languages_lowercase = train_languages_lowercase[czech_indices]
    elif args.notadditalian:
        train_languages = train_languages[italian_indices]
        train_languages_lowercase = train_languages_lowercase[italian_indices]

    for lan, lan_l in zip(train_languages, train_languages_lowercase):
        training_tasks.append(get_language_dataset(
            lan, lan_l, seed=args.seed, support_set_size=args.support_set_size))

    # Setting parameters
    DOING_MAML = args.maml
    if DOING_MAML or args.addenglish:
        # Get another training task
        training_tasks.append(
            get_language_dataset(
                "UD_English-EWT",
                "en_ewt-ud",
                seed=args.seed,
                support_set_size=args.support_set_size,
            )
        )
    UPDATES = args.updates
    EPISODES = args.episodes
    INNER_LR_DECODER = args.inner_lr_decoder
    INNER_LR_BERT = args.inner_lr_bert
    META_LR_DECODER = args.meta_lr_decoder
    META_LR_BERT = args.meta_lr_bert
    SKIP_UPDATE = args.skip_update

    # Filenames
    MODEL_FILE = (
        args.model_dir
        if args.model_dir is not None
        else (
            "../backup/pretrained/english_expmix_deps_seed2/2020.07.30_18.50.07"
            if not DOING_MAML
            else "logs/english_expmix_tiny_deps2/2020.05.29_17.59.31"
        )
    )
    train_params = get_params("metalearning", args.seed)

    m = Model.load(
        train_params,
        MODEL_FILE,
    )

    maml_string = "saved_models/MAML" if DOING_MAML else "saved_models/XMAML"
    param_list = [
        str(z)
        for z in [
            maml_string,
            INNER_LR_DECODER,
            INNER_LR_BERT,
            META_LR_DECODER,
            META_LR_BERT,
            UPDATES,
            args.seed,
        ]
    ]
    MODEL_SAVE_NAME = "_".join(param_list)
    MODEL_VAL_DIR = MODEL_SAVE_NAME + args.name
    META_WRITER = MODEL_VAL_DIR + "/meta_results.txt"

    if not os.path.exists(MODEL_VAL_DIR):
        subprocess.run(["mkdir", MODEL_VAL_DIR])
        subprocess.run(["mkdir", MODEL_VAL_DIR + "/performance"])
        subprocess.run(["mkdir", MODEL_VAL_DIR + "/predictions"])
        subprocess.run(["cp", "-r", MODEL_FILE + "/vocabulary", MODEL_VAL_DIR])
        subprocess.run(["cp", MODEL_FILE + "/config.json", MODEL_VAL_DIR])

    with open(META_WRITER, "w") as f:
        f.write("Model ready\n")

    # Loading the model
    train_params = get_params("metalearning", args.seed)
    prepare_environment(train_params)
    m = Model.load(
        train_params,
        MODEL_FILE,
    )
    meta_m = MAML(
        m, INNER_LR_DECODER, INNER_LR_BERT, first_order=True, allow_unused=True
    ).cuda()
    optimizer = Adam(
        [
            {
                "params": meta_m.module.text_field_embedder.parameters(),
                "lr": META_LR_BERT,
            },
            {"params": meta_m.module.decoders.parameters(), "lr": META_LR_DECODER},
            {"params": meta_m.module.scalar_mix.parameters(), "lr": META_LR_DECODER},
        ],
        META_LR_DECODER,
    )  # , weight_decay=0.01)

    scheduler = get_cosine_schedule_with_warmup(optimizer, 50, 500)

    for iteration in range(EPISODES):
        iteration_loss = 0.0

        """Inner adaptation loop"""
        for j, task_generator in enumerate(training_tasks):
            learner = meta_m.clone()

            # Sample two batches
            support_set = next(task_generator)[0]
            if SKIP_UPDATE == 0.0 or torch.rand(1) > SKIP_UPDATE:
                for mini_epoch in range(UPDATES):
                    inner_loss = learner.forward(**support_set)["loss"]
                    learner.adapt(inner_loss, first_order=True)
                    del inner_loss
                    torch.cuda.empty_cache()
            del support_set

            query_set = next(task_generator)[0]

            eval_loss = learner.forward(**query_set)["loss"]
            iteration_loss += eval_loss

            del eval_loss
            del learner
            del query_set
            torch.cuda.empty_cache()

        # Sum up and normalize over all 7 losses
        iteration_loss /= len(training_tasks)
        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()
        scheduler.step()

        # Bookkeeping
        with torch.no_grad():
            print(iteration, "meta", iteration_loss.item())
            with open(META_WRITER, "a") as f:
                f.write(str(iteration) + " meta " + str(iteration_loss.item()))
                f.write("\n")
        del iteration_loss
        torch.cuda.empty_cache()

        if iteration + 1 in [500, 1500, 2000] and not (
            iteration + 1 == 500 and DOING_MAML
        ):
            backup_path = os.path.join(
                MODEL_VAL_DIR, "model" + str(iteration + 1) + ".th"
            )
            torch.save(meta_m.module.state_dict(), backup_path)

    print("Done training ... archiving three models!")
    for i in [500, 600, 900, 1200, 1500, 1800, 2000, 1500]:
        filename = os.path.join(MODEL_VAL_DIR, "model" + str(i) + ".th")
        if os.path.exists(filename):
            save_place = MODEL_VAL_DIR + "/" + str(i)
            subprocess.run(["mv", filename, MODEL_VAL_DIR + "/best.th"])
            subprocess.run(["mkdir", save_place])
            archive_model(
                MODEL_VAL_DIR,
                files_to_archive=train_params.files_to_archive,
                archive_path=save_place,
            )
    subprocess.run(["rm", MODEL_VAL_DIR + "/best.th"])


if __name__ == "__main__":
    main()
