# Model-Based Robust Deep Learning (MBRDL)

In this repository, we include the code necessary for reproducing the code used in [Model-Based Robust Deep Learning](https://arxiv.org/abs/2005.10247).  In particular, we include the code necessary for both training models of natural variation as well as the code needed to train classifiers using these learned models.  A brief summary of the functionality provided in this repo is provided below in the table of contents.  If you find this repository useful in your research, please consider citing:

```latex
@article{robey2020model,
  title={Model-Based Robust Deep Learning},
  author={Robey, Alexander and Hassani, Hamed and Pappas, George J},
  journal={arXiv preprint arXiv:2005.10247},
  year={2020}
}
```

## Table of contents

First, we given instructions for how to setup the appropriate environment for this repository.
* [Setup instructions](#setup-instructions)

Next, we give details about how to train classifiers using the MBRDL paradigm.  ur implementation is based on the [Lambda Labs implementation](https://github.com/lambdal/imagenet18) of the ImageNet training repository.  

* [Training classifiers using MAT, MRT, and MDA](#training-classifiers-in-the-mbrdl-paradigm)  
    1. [Dataset selection](#dataset-selection)
    2. [Architecture and hyperparameters](#architecture-and-hyperparameters)
    3. [Using pre-trained models of natural variation](#using-pre-trained-models-of-natural-variation)
    4. [Composing models of natural variation](#composing-models-of-natural-variation)
    5. [Training algorithms](#training-algorithms)
    6. [Distributed settings](#distributed-settings)

Following this, we give a high-level overview of the structure of this repository, as well as an overview of the scripts that are included for training models of natural variation and classifiers that are robust to natural variation.

* [An overview of the structure of this repository](#an-overview-of-the-structure-of-this-repository)

We also provide code that can be used to train models of natural variation using the [MUNIT](https://arxiv.org/abs/1804.04732) framework.  The code that we use to train these models is largely based on the original [implementation of MUNIT](https://github.com/NVlabs/MUNIT).

* [Training models of natural variation](#training-models-of-natural-variation)
    1. [Retrieving a saved model of natural variation](#retrieving-a-saved-model-of-natural-variation)
    2. [Using other architectures for models of natural variation](#using-other-architectures-for-models-of-natural-variation)

In addition to providing functionality to train new models of natural variation, we also provide a library of pre-trained models of natural variation in a Google Drive folder.

* [A library of pre-trained models of natural variation](#a-library-of-pre-trained-models-of-natural-variation)

Finally, we identify common usage issues and provide solutions.

* [Trouble-shooting](#trouble-shooting)

## Setup instructions

After cloning this repository, the first step is to setup a virtual environment.

```bash
python3 -m venv mbrdl
source mbrdl/bin/activate
pip3 install -r requirements.txt
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
```

We also need to install NVIDIA's half-precision training tool [apex](https://github.com/NVIDIA/apex).  The setup instructions for `apex` are [here](https://github.com/NVIDIA/apex#quick-start).    

## An overview of the structure of this repository

In this repository, we provide four scripts that can be used to train models of natural variation as well as classifiers that are robust against natural variation; the scripts `train_munit.sh` and `train_munit_imagenet.sh` can be used to train models of natural variation with the MUNIT framework, and the scripts `train_basic.sh` and `train_imagenet.sh` can be used to train classifiers using the MBRDL paradigm.  

All of the components to train models of natural variation and classifiers is included in the `core/` directory.  In particular, `core/` is organized in the following way:

```bash
mbrdl/core/
├── classifiers
├── data
├── models
├── training
├── train_munit.py
├── train.py
└── utils
```

The `classifiers/` directory contains all code necessary to initialize different classifier architectures.  The `data/` directory contains dataloaders for MUNIT and for training classifiers for SVHN, CURE-TSR, GTSRB, and ImageNet/ImageNet-c.  The `models/` directory has code that can be used to load and train models of natural variation using the MUNIT framework.  The `training/` directory contains the training algorithms (including implementations of MAT, MRT, and MDA) and schedulers that are used to train classifiers.  The `utils/` directory has a variety of utilities that are used throughout the scripts that train classifiers in the MBRDL paradigm.

The two main training python files are `train_munit.py` and `train.py`.  As the name suggests, `train_munit.py` can be used to train models of natural variation using MUNIT.  `train.py` is a python file that trains classifiers in a distributed fashion.

## Training classifiers in the MBRDL paradigm

We also include the code needed to train classifiers that are robust to natural variation.  Our implementation is based on the [Lambda Labs implementation](https://github.com/lambdal/imagenet18) of the ImageNet training repository.  

To train a classifier on MNIST, SVHN, GTSRB, or CURE-TSR, you can run the following shell script:

```bash
chmod +x train_basic.sh
./train_basic.sh
```

Editing the `train_basic.sh` script will allow you to control the dataset, optimization parameters, model of natural variation, classifier architecture, and other hyperparameters.  To train on ImageNet/ImageNet-c, we provide a separate script, which can be run in the following way:

```bash
chmod +x train_imagenet_script.sh
./train_imagenet.sh
```

In what follows, we describe different settings for thes files.

### Dataset selection

By editing flags at the beginning of the file, you can change the dataset and the source of natural variation that are used for training/testing the classifier.  For example, to run with MNIST, you can set:

```bash
export DATASET='mnist'
export TRAIN_DIR=./datasets/MNIST/mnist
export SOURCE='background-color'
```

To run with SVHN, you can set:

```bash
export DATASET='svhn'
export TRAIN_DIR=./datasets/svhn
export SOURCE='brightness'
```

The choices for `SOURCE` for SVHN are choices are 'brightness', 'contrast', and 'contrast+brightness'.  The same choices are also available for GTSRB:

```bash
export DATASET='gtsrb'
export TRIAIN_DIR=./datasets/gtsrb/GTSRB
export SOURCE='brightness'
```

For CURE-TSR, you can select any of the sources of natural variation listed in the [original repository for CURE-TSR](https://github.com/olivesgatech/CURE-TSR#challenging-conditions) (e.g. snow, rain, haze, decolorization, etc.).  For example, you can set

```bash
export DATASET='cure-tsr'
export TRAIN_DIR=./datasets/cure_tsr/raw_data
export SOURCE='snow'
```

To train with ImageNet, you need to set the `TRAIN_DIR` and `VAL_DIR` flags depending on the install location of the ImageNet dataset.  This is to allow you to train with ImageNet and then evaluate on ImageNet-c.  

```bash
export DATASET='imagenet'
export TRAIN_DIR=./datasets/imagenet/train
export VAL_DIR=./datasets/imagenet-c/weather/snow/3
```

Note that when training with ImageNet, the `SOURCE` flag will not be used.



### Architectures, hyperparameters, and optimizers

To select the classifier architecture, you can set the following flags:

```bash
export ARCHITECTURE='basic'
export N_CLASSES=10         # number of classes
export SZ=32                # dataset image size (SZ x SZ x 3)
export BS=64                # batch size
```

The 'basic' architecture is a simple CNN with two convolutional layers and two feed-forward layers.  The program will also accept any of the architectures in [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html), including AlexNet and ResNet50.  For example, an appropriate ImageNet configuration could be something like this:

```bash
export ARCHITECTURE='resnet50'
export N_CLASSES=1000           # number of classes
export SZ=224                   # dataset image size (SZ x SZ x 3)
export BS=64                    # batch size
```

These flags will also allow you to set the number of output classes for the given architecture, the size of the images in the dataset, and the (training) batch size.  

You can also select the optimizer to be used for training the given `ARCHITECTURE`.  Currently, two optimizers are supported: SGD with momentum and [AdaDelta](https://arxiv.org/abs/1212.5701).  In our paper, we used SGD for the experiments on ImageNet, and AdaDelta for every other experiment (e.g. MNIST, SVHN, CURE-TSR, GTSRB, etc.).  To select an optimizer, you can use the `--optimizer` flag, which currently supports the arguments `sgd` or `adadelta`.  

### Using pre-trained models of natural variation

Yu can set the path to a saved model of natural variation and the dimension of the nuisance space Δ by setting

```bash
export MODEL_PATH=./core/models/learned_models/svhn-brightness.pt
export CONFIG_PATH=./core/models/munit/munit.yaml
export DELTA_DIM=8
```

The `CONFIG_PATH` should point to a `.yaml` file with appropriate settings for the MUNIT architecture.  An example is given in `core/models/munit/munit.yaml`.  Note that the dimension must match the `style_dim` parameter in core/models/munit/munit.yaml if you are using the MUNIT framework.

### Composing models of natural variation

To compose two models of natural variation, you can simply pass multiple paths after the `--model-paths` argument.  For example, to compose models of contrast and brightness for SVHN, first set

```bash
export MODEL_PATH_1=./core/models/learned_models/svhn-brightness.pt
export MODEL_PATH_2=./core/models/learned_models/svhn-contrast.pt
```

and then add `--model-paths $MODEL_PATH_1 $MODEL_PATH_2` to the python command at the bottom of `train_bash.sh` or `train_imagenet.sh`.


### Training algorithms

By default, the script will train a classifier with the standard ERM formulation.  However, by adding flags, you can train classifiers using the three model-based algorithms from our paper (MAT, MRT, and MDA) as well as PGD.  For example, to train with MRT and k=10, you can add the flags `--mrt -k 10` to the `python -m torch.distributed.launch ...` command at the bottom of the file.  By replacing `--mrt` with `--mat` or `--mda`, you can change the algorithm to MAT or MDA respectively.  Similarly, you can use the `--pgd` flag to train with the [PGD algorithm](https://arxiv.org/abs/1706.06083).  By default, PGD runs with a step size of 0.01, ɛ=8/255, and 20 steps of gradient ascent.

### Distributed settings

You can set the distributed settings with the following flags:

```bash
export N_GPUS_PER_NODE=4
export N_NODES=1
```

This will control how your training is distributed (see [torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) and [torch.distributed.launch](https://pytorch.org/docs/stable/distributed.html#launch-utility)).


## Training models of natural variation

In this work, we used the [MUNIT](https://arxiv.org/abs/1804.04732) framework to learn models of natural variation.  The code that we use to train these models is largely based on the original [implementation of MUNIT](https://github.com/NVlabs/MUNIT).  To train a model of natural variation with MUNIT, you can run the following shell script:

```bash
chmod +x train_munit.sh
./train_munit.sh
```

You can change the dataset and various directories using the flags in `train_munit.sh`.  In particular, you can set the `DATASET` and `SOURCE` environmental variables in the same was as in `train_basic.sh` and `train_imagenet.sh`.  You can also set various paths, such as the path to the MUNIT configuration file and to directory where you would like to save your output:

```bash
export CONFIG_PATH=core/models/munit/munit.yaml
export OUTPUT_PATH=core/models/munit/results
```

The `CONFIG_PATH` should point to a `.yaml` file with appropriate settings for the MUNIT architecture.  An example is given in `core/models/munit/munit.yaml`.  Note that the parameter `style_dim` in this file corresponds to the dimension that will be used for the nuisance space Δ.  By default, we have set this to 8, which was the dimension used throughout the experiments section of our paper.

### Retrieving a saved model of natural variation

After running `train_munit.sh`, you can retrieve a saved model of natural variation from `${OUTPUT_PATH}/outputs/munit/checkpoint/gen_<xxxx>.pt`, where `<xxxx>` denotes the iteration number.  A module that can be used to reload this `.pt` file has been provided in the `MUNITModelOfNatVar` class in `core/models/load.py`.  This class can be used to instantiate a model of natural variation G(x, δ) with a forward pass that takes a batch of images and an appropriately sized nuisance parameter δ.  For example, running

```python
G = MUNITModelOfNatVar(args.model_path, reverse=False).cuda()
```

will return a model of natural variation that can be called in the following way:

```python
imgs, target = next(iter(data_loader))
delta = torch.randn(imgs.size(0), delta_dim, 1, 1).cuda()
mb_images = G(imgs.cuda(), delta)
```

Here, `mb_images` will be a batch of images that look semantically similar to `imgs` but will have different levels of natural variation.  Note that `delta_dim` must be set appropriately in this code snippet to match the `style_dim` parameter from the `.yaml` file located at `OUTPUT_PATH`.  

As MUNIT learns mappings in both directions (e.g. from domain A-->B and from domain B--> A), we use the `reverse` flag to control which direction the MUNIT model maps.  By default, `reverse=False`, meaning that G will map from domain A to B.  If `reverse=True`, G will map from domain B to A.

### Using other architectures for models of natural variation

To use other architectures for G, you can simply replace the `MUNITModelOfNatVar` instantiation in the `load_model` function in `core/models/load.py`.  In particular, the only requirement is that a model of natural variation should be instantiated as a `torch.nn.Module` with a forward pass function `forward` that takes as input a batch of images and a suitably sized nuisance parameter, i.e.

```python
import torch.nn as nn

class MyModel(nn.Module):
  def __init__(self, fname): 
    self.model = self.load_model(fname)

  def forward(x, delta):
    return self.model(x, delta)

  def load_model(fname):
    # Load model from file and return
    return
```

## A library of pre-trained models of natural variation

We provide a library of pre-trained models of natural variation in a public Google Drive folder.  In particular, this folder contains models for MNIST, SVHN, GTSRB, CURE-TSR and ImageNet/ImageNet-c:

```
https://drive.google.com/drive/folders/1J2OQSWuH4VQwWkHDTz_3mM7fQI2Iws6o?usp=sharing
```

We plan to update this folder frequently with more models of natural variation.

## Trouble-shooting

If you run the code in distributed mode over multiple GPUs, you may encounter errors after exiting a program via `CTRL-C`.  Often, this will result in an error that looks something like this:

```
RuntimeError: Address already in use
```

If this happens, processes on one or several of your GPUs may still be running.  You can check this by running something like

```bash
nvidia-smi
```

If this shows any running processes, you can kill them individually by their PID number.  Alternatively, you can kill all running python processes by running

```bash
pkill -9 python
```
