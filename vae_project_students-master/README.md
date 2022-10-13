# Variational Autoencoders

This repository contains the framework of the lab regarding Variational Autoencoders.

## Prerequisites

### packages

Install the required packages using ```pip install -r requirements.txt```. It is recommended to do this within a virtual
python environment (such as anaconda).

### wandb

Create an account on [Weights and Biases](https://wandb.ai/home). You'll need to create a new project. Call it
VAE_project. Go to the lines containing ``wandb.init(project="VAE_project", entity="YOURNAME", name=...)`` and change ``YOURNAME``
to your wandb user name. This line can be found in:

- ``dataset.py``
- ``test_ae.py``
- ``test_vae.py``
- ``train_ae.py``
- ``train_vae.py``

When running one of these files for the first time, wandb will ask you to log in.

## Brief file summary

You will find multiple files in the repository, which are explained below.

``dataset.py``: this class can load and visualise two datasets (MNIST digits and CIFAR10). You will mostly use MNIST
digits throughout the lab, but feel free to experiment with the CIFAR10 dataset.

``denormalize.py``: Training and test data is normalized before feeding it to the model. This class denomalizes the data
back. You will not need to use this class explicitly.

``losses.py``: this class serves to implement your loss functions.

``models.py``: this class contains all your PyTorch modules (deep networks). It is advised to go through them all. For
more info, see section “Lab”.

``options.py``: a class containing all your options for training/testing your networks. You are also allowed to put all
your hyperparameters here.

``test_ae.py``: this file will use the autoencoder to reconstruct images, generate new images and display the latent
space.

``test_dataset.py``: load some example images of the selected dataset in options.py. It will also print the shape of the
training set tensor, which can be useful for building your networks.

``test_vae``: same as test_ae.py, but for VAEs.

``train_ae``: a script to train an autoencoder.

``train_vae``: same as train_ae.py, but for VAEs.

``utils``: helper functions are defined here; it is advised to go through them.

## Explanation training methodology

Select a name for each model you train using ``--name modelname`` (e.g. "AE_01"). Once training is completed the model
is saved in the ``--save_path path`` directory (experiments/ by default), together with its config file. When evaluating
the model the config file is automatically read and used to load the model. Results during both training and evaluation
are visualised on [Weights and Biases](https://wandb.ai/home).

