import torch
import pickle
import numpy as np
import torch.utils.data as data_utils
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils.evaluation import evaluate_vae as evaluate
import matplotlib.gridspec as gridspec
import sys
from PIL import Image
import os

def plot_images(x_sample, dir, input_size=(1, 28, 28), input_type="binary"):

    for i, sample in enumerate(x_sample):
        fig = plt.figure(figsize=(1/2.75, 1/2.75))
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        sample = sample.reshape(input_size)
        sample = sample.swapaxes(0, 2)
        sample = sample.swapaxes(0, 1)
        if input_type == 'binary' or input_type == 'gray':
            sample = sample[:, :, 0]
            plt.imshow(sample, cmap='gray')
        else:
            plt.imshow(sample)
        print("image saved: {}{}".format(dir, i))
        plt.savefig("{}{}.png".format(dir, i), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

path = sys.argv[1]
folder = sys.argv[2]

BATCH_SIZE = 100

model = torch.load(path)
model.eval()

samples_x = model.generate_x(500)
input_size = [1, 28, 20]

if not os.path.exists("/content/vae_vampprior/output/{}/".format(folder)):
    os.makedirs("/content/vae_vampprior/output/{}/".format(folder))
plot_images(samples_x.cpu().detach().numpy(), "/content/vae_vampprior/output/{}/".format(folder), input_size)
