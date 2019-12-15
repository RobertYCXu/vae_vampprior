# CS480 Final Project: Evaluating and comparing VAE with VampPrior and standard Gaussian prior
This repository contains the code, trained models, and resulting generated images used for our
CS480 final project at the University of Waterloo. Code for generating and training different
models based on the VampPrior and standard Gaussian prior is based on the following paper:
* Jakub M. Tomczak, Max Welling, VAE with a VampPrior, [arXiv preprint](https://arxiv.org/abs/1705.07120), 2017

## Requirements
The code is compatible with:
* `pytorch 0.2.0`

## Data
The experiments can be run on the following datasets:
* static MNIST: links to the datasets can found at [link](https://github.com/yburda/iwae/tree/master/datasets/BinaryMNIST);
* binary MNIST: the dataset is loaded from PyTorch;
* OMNIGLOT: the dataset could be downloaded from [link](https://github.com/yburda/iwae/blob/master/datasets/OMNIGLOT/chardata.mat);
* Caltech 101 Silhouettes: the dataset could be downloaded from [link](https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat).
* Frey Faces: the dataset could be downloaded from [link](https://github.com/y0ast/Variational-Autoencoder/blob/master/freyfaces.pkl).
* Histopathology Gray: the dataset could be downloaded from [link](https://github.com/jmtomczak/vae_householder_flow/tree/master/datasets/histopathologyGray);
* CIFAR 10: the dataset is loaded from PyTorch.
* Fashion MNIST: the dataset is loaded from PyTorch.

## Run the experiment
1. Set-up your experiment in `experiment.py`.
2. Run experiment:
```bash
python experiment.py
```
## Models
You can run a vanilla VAE, a one-layered VAE or a two-layered HVAE with the standard prior or the VampPrior by setting `model_name` argument to either: (i) `vae` or `hvae_2level` for MLP, (ii) `convvae_2level` for convnets, (iii) `pixelhvae_2level` for (ii) with a PixelCNN-based decoder, and specifying `prior` argument to either `standard` or `vampprior`.

The trained models used and evaluated in our project can be found in the `trained_models`
directory. Specifically, models for the mnist, frey faces, cifar10, and fashion mnist datasets were
trained. Each model uses either a VampPrior or standard prior built on a vanilla VAE, two-layered
HVAE, 2 level convolutional network, or 2 level PixelCNN-based decoder.

## Quantitative Evaluations
Scripts for generating the FID (Frechet Inception Distance) and IS (Inception Score) for the
trained models can be found under the `evaluations` directory.

The metrics for each trained model (KL divergence, Log Likelihood, ELBO, IS, FID) can be found in `metric.txt`.

## Results
The generated images produced from our trained models can be found under the `generate` directory.
The script `generate_fake_images.py` was used to generate images from the trained models.

## Citation

Code used in this project is based on the paper mentioned above and should be appropriately cited
as such:

```
@article{TW:2017,
  title={{VAE with a VampPrior}},
  author={Tomczak, Jakub M and Welling, Max},
  journal={arXiv},
  year={2017}
}
```
