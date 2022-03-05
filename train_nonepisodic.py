"""This code performs multi-task learning in a non-episodic manner"""
import json
import subprocess
import argparse
import os
import torch
from torch import autograd
import numpy as np
from torch.optim import Adam

from allennlp.models.model import Model
from allennlp.models.archival import archive_model
from allennlp.common.util import prepare_environment
from udify import util


from get_language_dataset import get_language_dataset
from get_default_params import get_params
from naming_conventions import train_languages, train_languages_lowercase
from schedulers import get_cosine_schedule_with_warmup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=9999, type=int, help="Set seed")
    parser.add_argument("--lr_decoder", default=None,
                        type=float, help="Adaptation LR")
    parser.add_argument(
        "--lr_bert", default=None, type=float, help="Adaptation LR for BERT layers"
    )
    parser.add_argument("--episodes", default=600,
                        type=int, help="Episode amount")
    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        help="Directory from which to start training",
    )
    parser.add_argument("--support_set_size", default=None,
                        type=int, help="Batch size")
    parser.add_argument("--name", default="", type=str, help="Extra name")

    parser.add_argument("--addenglish", default=False,
                        type=bool, help="Add English as a task")
    parser.add_argument("--notaddhindi", default=False,
                        type=bool, help="Skip Hindi as a task")
    parser.add_argument("--notadditalian", default=False,
                        type=bool, help="Skip Italian as a task")
    parser.add_argument("--notaddczech", default=False,
                        type=bool, help="Skip Czech as a task")

    args = parser.parse_args()

    training_tasks = []
    train_languages = np.array(train_languages)
    train_languages_lowercase = np.array(train_languages_lowercase)
    hindi_indices = [0, 1, 2, 3, 4, 6]
    italian_indices = [0, 2, 3, 4, 5, 6]
    czech_indices = [1, 2, 3, 4, 5, 6]
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

    if args.addenglish:
        # Get another training task
        training_tasks.append(
            get_language_dataset(
                "UD_English-EWT",
                "en_ewt-ud",
                seed=args.seed,
                support_set_size=args.support_set_size,
            )
        )

    train_params = get_params("finetuning", args.seed)
    prepare_environment(train_params)

    EPISODES = args.episodes
    LR_DECODER = args.lr_decoder
    LR_BERT = args.lr_bert

    MODEL_SAVE_NAME = (
        "saved_models/finetune_"
        + str(LR_DECODER)
        + "_"
        + str(LR_BERT)
        + "_"
        + str(args.seed)
    )
    MODEL_VAL_DIR = MODEL_SAVE_NAME + args.name
    MODEL_FILE = (
        args.model_dir
        if args.model_dir is not None
        else "logs/english_expmix_deps/2020.05.17_01.08.52/"
    )

    if not os.path.exists(MODEL_VAL_DIR):
        subprocess.run(["mkdir", MODEL_VAL_DIR])
        subprocess.run(["mkdir", MODEL_VAL_DIR + "/performance"])
        subprocess.run(["mkdir", MODEL_VAL_DIR + "/predictions"])
        subprocess.run(["cp", "-r", MODEL_FILE + "/vocabulary", MODEL_VAL_DIR])
        subprocess.run(["cp", MODEL_FILE + "/config.json", MODEL_VAL_DIR])

    model = Model.load(train_params, MODEL_FILE).cuda()
    model.train()

    optimizer = Adam(
        [
            {"params": model.text_field_embedder.parameters(), "lr": LR_BERT},
            {"params": model.decoders.parameters(), "lr": LR_DECODER},
            {"params": model.scalar_mix.parameters(), "lr": LR_DECODER},
        ],
        LR_DECODER,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, 1000)
    with open(MODEL_VAL_DIR + "/losses.txt", "w") as f:
        f.write("model ready\n")
    losses = []

    for episode in range(EPISODES):
        for j, task in enumerate(training_tasks):
            input_set = next(task)[0]
            loss = model(**input_set)["loss"]
            # task_num_tokens_seen[j] += len(input_set['tokens']['tokens'][0])
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if episode + 1 in [1000, 1200, 1800]:
            backup_path = os.path.join(
                MODEL_VAL_DIR, "model" + str(episode + 1) + ".th"
            )
            torch.save(model.state_dict(), backup_path)

    for x in losses:
        with open(MODEL_VAL_DIR + "/losses.txt", "a") as f:
            f.write(str(x))
            f.write("\n")

    print("Done training ... archiving three models! ")
    for i in [600, 1000, 1200, 1800]:
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
