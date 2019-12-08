import torch
import pickle
import numpy as np
import torch.utils.data as data_utils
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils.evaluation import evaluate_vae as evaluate
import matplotlib.gridspec as gridspec
import sys

def plot_images(args, x_sample, dir, file_name, size_x=3, size_y=3, input_size=(1, 28, 28), input_type="binary"):

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

    plt.savefig(dir + file_name + '.png', bbox_inches='tight')
    plt.close(fig)

path = sys.argv[0]

BATCH_SIZE = 100

model = torch.load("/content/vae_frey_faces_standard_vae.model")
model.eval()

path = "/content/vae_vampprior/"
input_size = [1, 28, 20]

# start processing
with open('datasets/Freyfaces/freyfaces.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

data = (data[0] + 0.5) / 256.

x_test = data.reshape(-1, 28*20)

# pytorch data loader
test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
test_loader = data_utils.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

for batch_idx, (data, target) in enumerate(test_loader):
    try:
        data, target = data.cuda(), target.cuda()
    except:
        continue

    data, target = Variable(data), Variable(target)

    x = data

    if not os.path.exists(path + '/test1/'):
        os.makedirs(path + '/test1/')

    if not os.path.exists(dir + '/test2/'):
        os.makedirs(path + '/test2/')

    plot_images(args, data.data.cpu().numpy()[0:9], dir + 'test1/', 'real{}'.format(batch_idx), 3, 3, input_size)
    x_mean = model.reconstruct_x(x)
    plot_images(args, x_mean.data.cpu().numpy()[0:9], dir + 'test2/', 'fake{}'.format(batch_idx), 3, 3, input_size)

