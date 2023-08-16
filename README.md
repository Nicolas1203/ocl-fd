# Learning Representations on the Unit Sphere: Application to Online Continual Learning

This repository is the official implementation of our paper "Learning Representations on the Unit Sphere: Application to Online Continual Learning". This paper is currently under review.

# Method illustration

Our techniques fixes mean direction on the unit sphere for continual training like illustrated here.
![](illustration.png)

# Project structure

```bash
.
├── LICENSE
├── README.md
├── config
│   ├── all                 # All config files used to train the methods are here
│   └── parser.py           # Argument parser
├── illustration.png
├── main.py                 # Entrypoint of the project
├── requirements.txt        
├── results                 # results will be stored here as .csv
├── src                     # all the useful code is here
│   ├── buffers             
│   │   ├── buffer.py
│   │   ├── logits_res.py
│   │   └── reservoir.py
│   ├── datasets
│   ├── learners            # All methods are implemented here and inherit from base.py
│   │   ├── base.py
│   │   ├── baselines
│   │   │   ├── agem.py
│   │   │   ├── derpp.py
│   │   │   ├── dvc.py
│   │   │   ├── er.py
│   │   │   ├── er_ace.py
│   │   │   ├── gdumb.py
│   │   │   ├── lump.py
│   │   │   ├── ocm.py
│   │   │   └── scr.py
│   │   ├── ce.py
│   │   └── fd.py
│   ├── models
│   │   └── resnet.py
│   └── utils
│       ├── data.py
│       ├── losses.py       # Losses are implemented here
│       ├── metrics.py
│       ├── name_match.py   # Correspondance between learner, buffer and naming conventions
│       ├── tensorboard.py
│       └── utils.py
└── sweeps                  # Sweep configurations for HP search
```

# Installation

It is recommended to use a virtualenv or docker to run this code. Corresponding dependencies can be found in `requirements.txt.` In you use pip you can run `pip install -r requirements.txt` to install required packages.

# Usage

Command line usage of the current repository is described here.

```bash
usage: main.py [-h] [--config CONFIG] [--train] [--test] [--epochs N]
               [--start-epoch N] [-b BATCH_SIZE] [--learning-rate LEARNING_RATE]
               [--momentum M] [--weight-decay W] [--optim {Adam,SGD}] [--save-ckpt]
               [--seed SEED] [--tag TAG] [--tb-root TB_ROOT] [--logs-root LOGS_ROOT]
               [--results-root RESULTS_ROOT] [--tensorboard] [--verbose]
               [--ckpt-root CKPT_ROOT] [--resume] [--model-state MODEL_STATE]
               [--buffer-state BUFFER_STATE] [--head HEAD] [--proj-dim PROJ_DIM]
               [--nb-channels NB_CHANNELS] [--eval-proj] [--pretrained]
               [--supervised] [--dim-int DIM_INT] [-nf NF]
               [--data-root-dir DATA_ROOT_DIR] [--min-crop MIN_CROP]
               [--dataset {mnist,fmnist,cifar10,cifar100,tiny,sub,yt}]
               [--training-type {uni,inc,blurry}] [--n-classes N_CLASSES]
               [--img-size IMG_SIZE] [--num-workers NUM_WORKERS] [--n-tasks N_TASKS]
               [--labels-order LABELS_ORDER [LABELS_ORDER ...]]
               [--blurry-scale BLURRY_SCALE] [--temperature T] [--mem-size MEM_SIZE]
               [--mem-batch-size MEM_BATCH_SIZE] [--buffer BUFFER]
               [--drop-method {random}] [--mem-iters MEM_ITERS] [--learner LEARNER]
               [--debug] [--eval-mem] [--eval-random] [--lab-pc LAB_PC]
               [--n-runs N_RUNS] [--start-seed START_SEED] [--run-id RUN_ID]
               [--kornia] [--no-kornia] [--n-augs N_AUGS] [--tf-type {full,partial}]
               [--var VAR] [--fd-loss FD_LOSS] [-mu MU] [--derpp-alpha DERPP_ALPHA]
               [--derpp-beta DERPP_BETA] [--no-wandb] [--wandb-watch] [--sweep]

Pytorch UCL, with tensorboard logs.

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the configuration file for the training to launch.
  --train
  --test
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        mini-batch size (default: 10)
  --learning-rate LEARNING_RATE, -lr LEARNING_RATE
                        Initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 0)
  --optim {Adam,SGD}
  --save-ckpt           whether to save chekpoints or not
  --seed SEED           Random seed to use.
  --tag TAG, -t TAG     Base name for graphs and checkpoints
  --tb-root TB_ROOT     Where do you want tensorboards graphs ?
  --logs-root LOGS_ROOT
                        Defalt root folder for writing logs.
  --results-root RESULTS_ROOT
                        Where you want to save the results ?
  --tensorboard
  --verbose
  --ckpt-root CKPT_ROOT
                        Directory where to save the model.
  --resume, -r          Resume old training. Setup model state and buffer state.
  --model-state MODEL_STATE
  --buffer-state BUFFER_STATE
  --head HEAD
  --proj-dim PROJ_DIM
  --nb-channels NB_CHANNELS
                        Number of channels for the input image.
  --eval-proj           Use projection for inference. (default is representation.)
  --pretrained          Use a pretrained model if available.
  --supervised          Pseudo labels or true labels ?
  --dim-int DIM_INT
  -nf NF                Number of feature for Resnet18. Set nf=20 for reduced
                        resnet18, nf=64 for full.
  --data-root-dir DATA_ROOT_DIR
                        Root dir containing the dataset to train on.
  --min-crop MIN_CROP   Minimum size for cropping in data augmentation. range (0-1)
  --dataset {mnist,fmnist,cifar10,cifar100,tiny,sub,yt}, -d {mnist,fmnist,cifar10,cifar100,tiny,sub,yt}
                        Dataset to train on
  --training-type {uni,inc,blurry}
                        How to feed the data to the network (incremental context or
                        not)
  --n-classes N_CLASSES
                        Number of classes in database.
  --img-size IMG_SIZE   Size of the square input image
  --num-workers NUM_WORKERS, -w NUM_WORKERS
                        Number of workers to use for dataloader.
  --n-tasks N_TASKS     How many tasks do you want ?
  --labels-order LABELS_ORDER [LABELS_ORDER ...]
                        In which order to you want to see the labels ? Random if not
                        specified.
  --blurry-scale BLURRY_SCALE
  --temperature T       temperature parameter for softmax
  --mem-size MEM_SIZE   Memory size for continual learning
  --mem-batch-size MEM_BATCH_SIZE, -mbs MEM_BATCH_SIZE
                        How many images do you want to retrieve from the memory/ltm
  --buffer BUFFER       What buffer do you want ? See available buffers in
                        utils/name_match.py
  --drop-method {random}
                        How to drop images from memory when adding new ones.
  --mem-iters MEM_ITERS
                        Number of iterations on memory
  --learner LEARNER     What learner do you want ? See list of available learners in
                        utils/name_match.py
  --debug
  --eval-mem
  --eval-random
  --lab-pc LAB_PC       Number of labeled images per class to use in unsupervised
                        evaluation.
  --n-runs N_RUNS       Number of runs, with different seeds each time.
  --start-seed START_SEED
                        First seed to use.
  --run-id RUN_ID       Id of the current run in multi run.
  --kornia
  --no-kornia
  --n-augs N_AUGS
  --tf-type {full,partial}
                        Data augmentation sequence to use.
  --var VAR             Variance for fd loss. kappa=1/var for VMF and kappa=1/2*var
                        for AGD.
  --fd-loss FD_LOSS     vmf or agd loss
  -mu MU                mu value for FD model.
  --derpp-alpha DERPP_ALPHA
                        Values of alpha un der++ loss
  --derpp-beta DERPP_BETA
                        Values of beta un der++ loss
  --no-wandb
  --wandb-watch         Watch the models gradient and parameters into wandb (can be
                        slow).
  --sweep               Run the code with sweep for HP search.
```

# Training

## Command line only

Training can be done by specifying parameters in command line, for example:

    python main.py --results-root ./results/tiny/ --data-root /data/dataset/torchvision --learner FD --dataset tiny --batch-size 10 --fd-loss agd --optim Adam --learning-rate 0.0005

## Using a configuration file (recommended)

When using a configuration file, parameters specified in the .yaml cannot be overriten by command line arguments. However, other parameters like `--data-root` can be adapted to the users' need.

    python main.py --data-root /Volumes/SSD2/data/dataset/torchvision --config config/all/ER,cifar10,m1000mbs64sbs10,blurry1500.yaml

## output example

Output of the command above should contain performances like this.

    root - INFO - --------------------FORGETTING--------------------
    root - INFO - ncm     0.0000   nan      nan      nan      nan      0.0000
    root - INFO - ncm     0.2885   0.0000   nan      nan      nan      0.2885
    root - INFO - ncm     0.2935   0.2225   0.0000   nan      nan      0.2580
    root - INFO - ncm     0.4615   0.3190   0.0370   0.0000   nan      0.2725
    root - INFO - ncm     0.5815   0.2155   0.1795   0.0250   0.0000   0.2504
    root - INFO - --------------------ACCURACY--------------------
    root - INFO - ncm     0.7750   nan      nan      nan      nan      0.7750
    root - INFO - ncm     0.4865   0.5260   nan      nan      nan      0.5062
    root - INFO - ncm     0.4815   0.3035   0.5150   nan      nan      0.4333
    root - INFO - ncm     0.3135   0.2070   0.4780   0.2875   nan      0.3215
    root - INFO - ncm     0.1935   0.3105   0.3355   0.2625   0.3045   0.2813

# Datasets

#https://github.com/vlomonaco/core50# CIFAR10 and CIFAR100

These datasets are automatically downloaded if not present.

## Tiny-IN

TODO

## CORe50

Download `cropped_128x128_images.zip`from [the official website](https://vlomonaco.github.io/core50/index.html#download) or do run:

        wget http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip

then exatract and rename make the changes so that you have the following file architecture:

"""bash
core50/
├── test
│   ├── s10
│   ├── s3
│   └── s7
└── train
    ├── s1
    ├── s11
    ├── s2
    ├── s4
    ├── s5
    ├── s6
    ├── s8
    └── s9
"""

# Cite

Coming soon.

