# Meta-Learning for UD parsing
This codebase extends UDify with Model-Agnostic Meta-Learning. 

In order to do so, we modified these libraries slightly:

- `learn2learn` (Necessary code in `ourmaml.py`, which is the wrapper for MAML around UDify, adapted for different learning rates with SGD)

- `udify` (See directory, minor tweaks with utf-8 encoding and parsing dependency trees while loading data)

- `allennlp==0.9.0` (See directory, minor tweaks with utf-8 encoding + UDify model working together)

# Training process
## Pre-training
The code for pre-training can be found in `pretrain.py`. 
It is vital that the entire vocabulary is used during pre-training (that is, vocabulary of all languages you would later want to train or test on.). The treebanks of relevant languages can be concatenated using UDify's `concat_treebanks.py`.

After pre-training, the model directory can be passed to a meta-learner or a non-episodic learner: 

## Meta-Learning
All code for the meta-training process can be found in `train_meta.py`. 

## Non-Episodic Learning
All code for the meta-training process can be found in `train_nonepisodic.py`. 

# Evaluation 
There are multiple ways to evaluate the models.
* Meta-validation / Meta-testing

  This is defined in `metatest_all.py`.
* Zero-Shot (Not used for final paper)

  This can simply be done by passing a model directory to `zeroshot_test.py`

# Analysis
The code for non-projectiveness can be seen in `projective.py`.

# UDify and data
See the original UDify README for more info: https://github.com/Hyperparticle/udify

In particular, see the update to newer allennlp (which used pytorch dataloaders) from last summer (after all of our struggles) in this PR https://github.com/tamuhey/udify/pull/1, but I haven't looked at this yet, so this code still runs on the old allennlp version.

Data can be downloaded with script in `scripts` folder, or directly from https://universaldependencies.org/. 

# Examples 
An example of how to run the code when you have pretrained a model with `pretrain.py` and put it into directory `pretrained/pretrained_hindi`: 

`python train_meta.py --name hindi --notaddhindi True --addenglish True --inner_lr_decoder 0.0001 --inner_lr_bert 1e-05 --meta_lr_decoder 0.0007 --meta_lr_bert 1e-05 --updates 20 --episodes 500 --seed 19 --support_set_size 20 --model_dir pretrained/pretrained_hindi` 

This will save a model to the directory `saved_models/XMAML_0.0001_1e-05_0.0007_1e-05_20_19hindi`, we can meta-validate using this script:

`python3.7 metatest_all.py --validate True --lr_decoder 0.0001 --lr_bert 1e-05 --updates 20 --support_set_size 20 --optimizer sgd --seed 3 --episode 500 --model_dir saved_models/XMAML_0.0001_1e-05_0.0007_1e-05_20_19hindi`