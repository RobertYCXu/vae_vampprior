import torch
import pickle
import numpy as np
import torch.utils.data as data_utils
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils.evaluation import evaluate_vae as evaluate
import matplotlib.gridspec as gridspec
import sys

def plot_images(x_sample, dir, file_name, size_x=3, size_y=3, input_size=(1, 28, 28), input_type="binary"):

    fig = plt.figure(figsize=(size_x, size_y))
    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(x_sample):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        sample = sample.reshape(input_size)
        sample = sample.swapaxes(0, 2)
        sample = sample.swapaxes(0, 1)
        if input_type == 'binary' or input_type == 'gray':
            sample = sample[:, :, 0]
            plt.imshow(sample, cmap='gray')
        else:
            plt.imshow(sample)

    plt.savefig(path + file_name + '.png', bbox_inches='tight')
    plt.close(fig)

path = sys.argv[1]
folder = sys.argv[2]

BATCH_SIZE = 100

model = torch.load(path)
model.eval()

samples = model.generate_x(25)
input_size = list(samples[0].shape)

for i, data in enumerate(samples):
    plot_images(x_mean.data.cpu().numpy()[0:9], path + "/{}/".format(folder), 'fake{}'.format(i), 3, 3, input_size)
