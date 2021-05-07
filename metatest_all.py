"""
Meta-tests or meta-validates.
Samples args.amount support sets of size args.support_set_size,
     and passes them through the model args.updates times.
"""

from allennlp.models.archival import archive_model
from get_language_dataset import get_language_dataset
from get_default_params import get_params
from allennlp.models.model import Model
from allennlp.common.util import prepare_environment
from torch.optim import Adam, SGD
import torch
import subprocess
from udify import util
import os
from naming_conventions import (
    languages,
    languages_lowercase,
    validation_languages,
    validation_languages_lowercase,
)
from naming_conventions import (
    languages_too_small_for_20_batch_20,
    languages_too_small_for_20_batch_20_lowercase,
)
from get_language_dataset import get_language_dataset, get_test_set
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=9999, type=int, help="Set seed")
    parser.add_argument("--support_set_size", default=32, type=int, help="Support set size")
    parser.add_argument(
        "--start_from_pretrain", default=0, type=int, help="Whether to start from pretrain"
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        help="Directory from which to start testing if not starting from pretrain",
    )
    parser.add_argument(
        "--episode",
        default=None,
        type=int,
        help="Saved episode from which to start testing.",
    )
    parser.add_argument("--name", default=None, type=str, help="Backwards compatability")
    parser.add_argument(
        "--skip_update", default=None, type=float, help="Backwards compatability"
    )
    parser.add_argument(
        "--lr_decoder",
        default=None,
        type=float,
        help="Fast adaptation output learning rate for the decoder",
    )
    parser.add_argument(
        "--lr_bert",
        default=None,
        type=float,
        help="Fast adaptation output learning rate for BERT",
    )
    parser.add_argument(
        "--updates", default=3, type=int, help="Amount of inner loop updates"
    )
    parser.add_argument(
        "--optimizer", default="adam", type=str, help="Which optimizer to use? [adam|sgd]"
    )
    parser.add_argument(
        "--validate",
        default=False,
        type=bool,
        help="Meta-validate on validation language, for hyperparameter search",
    )
    parser.add_argument(
        "--amount",
        default=20,
        type=int,
        help="Amount of experiments to do /meta-test batches to sample",
    )
    parser.add_argument(
        "--batches", default=1, type=int, help="How many batches to sample per update"
    )

    args = parser.parse_args()

    # The model on which to Meta_test
    MODEL_DIR_PRETRAIN = "logs/english_expmix_deps/2020.05.17_01.08.52/"
    MODEL_DIR_FINETUNE = os.path.join(
        args.model_dir, str(args.episode) if args.episode is not None else ""
    )

    MODEL_DIR = MODEL_DIR_FINETUNE if args.start_from_pretrain == 0 else MODEL_DIR_PRETRAIN

    MODEL_NAMEDIR = (
        args.model_dir.replace("/", "-") + "_" + str(args.episode)
        if args.start_from_pretrain == 0
        else "ONLY"
    )
    VALIDATING = args.validate

    # Setting all the parameters
    LR_DECODER = args.lr_decoder
    LR_BERT = args.lr_bert
    UPDATES = args.updates
    BATCHES = args.batches  # * args.support_set_size

    # Extract
    subprocess.run(
        [
            "tar",
            "-x",
            "-z",
            "-v",
            "-f",
            os.path.join(MODEL_DIR, "model.tar.gz"),
            "-C",
            MODEL_DIR,
        ]
    )
    subprocess.run(
        ["mv", os.path.join(MODEL_DIR, "weights.th"), os.path.join(MODEL_DIR, "best.th")]
    )

    # Is it validation time, or test time?
    if args.validate:
        the_languages = validation_languages
        the_languages_lowercase = validation_languages_lowercase
        extra_string = "metavalidation"
    else:
        the_languages = languages
        the_languages_lowercase = languages_lowercase
        extra_string = "metatesting"

    paramlist = [
        extra_string,
        str(LR_DECODER),
        str(LR_BERT),
        str(UPDATES),
        str(args.support_set_size),
        args.optimizer,
        MODEL_NAMEDIR,
    ]
    WHERE_TO_SAVE = "_".join(paramlist)
    USE_ADAM = args.optimizer == "adam"

    print("Saving all to directory", WHERE_TO_SAVE)
    print("Running from", MODEL_DIR, "with learning rates", LR_DECODER, LR_BERT)
    subprocess.run(["mkdir", WHERE_TO_SAVE])

    # The languages on which to evaluate
    for i, language in enumerate(the_languages):
        test_file = get_test_set(
            language,
            the_languages_lowercase[i],
            number=0,
            validating=VALIDATING,
            bs=args.support_set_size * args.batches,
        )
        val_iterator = get_language_dataset(
            language,
            the_languages_lowercase[i],
            seed=2002,  # Seed should always be the same for validation iterators,
            # Since we might want to sample more times than the datasets permit, we want to shuffle
            # And the OUTER seed is already random, thus validating on the same 15-20 sets
            support_set_size=args.support_set_size,
            number=0,
            validate=True,
            bs=args.support_set_size * args.batches,
        )

        # Create directory and copy relevant files there for later
        SERIALIZATION_DIR = WHERE_TO_SAVE + "/resultsvalidation" + language
        # if os.path.exists(SERIALIZATION_DIR): continue
        # Try with 20 different batches from validation set.
        NO = 0
        for TRY in range(args.amount):

            subprocess.run(["mkdir", SERIALIZATION_DIR])
            subprocess.run(["cp", "-r", MODEL_DIR + "/vocabulary", SERIALIZATION_DIR])
            subprocess.run(["cp", MODEL_DIR + "/config.json", SERIALIZATION_DIR])

            # Set up model and iterator and optimizer
            train_params = get_params("metatesting", args.seed)
            prepare_environment(train_params)
            m = Model.load(train_params, MODEL_DIR).cuda()

            if USE_ADAM:
                optimizer = Adam(
                    [
                        {"params": m.text_field_embedder.parameters(), "lr": LR_BERT},
                        {"params": m.decoders.parameters(), "lr": LR_DECODER},
                        {"params": m.scalar_mix.parameters(), "lr": LR_DECODER},
                    ],
                    LR_DECODER,
                )
            else:
                optimizer = SGD(
                    [
                        {"params": m.text_field_embedder.parameters(), "lr": LR_BERT},
                        {"params": m.decoders.parameters(), "lr": LR_DECODER},
                        {"params": m.scalar_mix.parameters(), "lr": LR_DECODER},
                    ],
                    LR_DECODER,
                )

            for BATCH in range(BATCHES):
                try:
                    support_set = next(val_iterator)[0]
                except StopIteration:
                    test_file = get_test_set(
                        language,
                        the_languages_lowercase[i],
                        number=NO,
                        validating=VALIDATING,
                        bs=args.support_set_size * args.batches,
                    )
                    val_iterator = get_language_dataset(
                        language,
                        the_languages_lowercase[i],
                        number=NO,
                        seed=2002,  # Seed should always be the same for validation iterators,
                        # Since we might want to sample more times than the datasets permit, we want to shuffle
                        # And the OUTER seed is already random, thus validating on the same 15-20 sets
                        support_set_size=args.support_set_size,
                        validate=True,
                        bs=args.support_set_size * args.batches,
                    )
                    support_set = next(val_iterator)[0]
                NO += 1

                # Do one forward pass
                for mini_epoch in range(UPDATES):
                    loss = m.forward(**support_set)["loss"]
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    del loss

            # Get specific predictions for the model that has few-shot learned
            current_pred_file = os.path.join(
                WHERE_TO_SAVE, language + "_predictions" + str(TRY) + ".conllu"
            )
            current_output_file = os.path.join(
                WHERE_TO_SAVE, language + "_performance" + str(TRY) + ".json"
            )

            util.predict_model_without_archive(
                m,
                "udify_predictor",
                get_params("metatesting", args.seed),
                SERIALIZATION_DIR,
                test_file,
                current_pred_file,
                current_output_file,
                batch_size=16,
            )

            # Clean up
            print("Wrote", current_output_file, "removing", SERIALIZATION_DIR)
            subprocess.run(["rm", "-r", "-f", SERIALIZATION_DIR])

            del m
            del optimizer
        del val_iterator

if __name__ == "__main__":
    main()
