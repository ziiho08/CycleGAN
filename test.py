import argparse
import itertools
import sys
import os
import torch
import pandas as pd
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import *
from dataset import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="dataset_", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--folder_name", type=str, default="default_mask", help="name of the dataset")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1, help="interval between saving generator outputs")
parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument("--folder_num", type=int, default=10, help="number of the dataset")
parser.add_argument('--GPU_NUM', type=int, default=0, help='number of channels of output data')
parser.add_argument('--Save_name', type=str, default="try1")
parser.add_argument('--model_num', type=int, default=190)

opt = parser.parse_args()

print(opt)
GPU_NUM = opt.GPU_NUM

device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print('\t' * 3 + 'Current cuda device : ', torch.cuda.current_device())
print('\t' * 3 + f'{torch.cuda.get_device_name(GPU_NUM)}')

# Create sample and checkpoint directories
os.makedirs("test_images/%s" %opt.Save_name, exist_ok=True)
#os.makedirs("test_loss/%s" % opt.dataset_name, exist_ok=True)

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Network
G_AB = Generator(1, 64, opt.n_residual_blocks)
G_BA = Generator(1, 64, opt.n_residual_blocks)

if cuda:
    G_AB = G_AB.cuda()
    G_AB = G_BA.cuda()

# Load pretrained models
G_AB.load_state_dict(torch.load("saved_models/%s%s/G_AB_%d.pth" % (opt.dataset_name, opt.Save_name, opt.model_num)))
G_BA.load_state_dict(torch.load("saved_models/%s%s/G_BA_%d.pth" % (opt.dataset_name, opt.Save_name, opt.model_num)))

# Set model's test mode
G_AB.eval()
G_AB.eval()

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

test_dataloader = DataLoader(
    ImageDataset("./", folder_name = opt.folder_name,folder_num= opt.folder_num),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(test_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid,  "test_images/%s/%s.png" % (opt.Save_name, iter_done), normalize=False)

# ----------
#  Test
# ----------

for i, batch in enumerate(test_dataloader):

    # Set model input
    real_A = Variable(batch["A"][2000:].type(Tensor))
    real_B = Variable(batch["B"].type(Tensor))

    # Generator Output
    fake_B = 0.5*G_AB(real_A).data + 1.0
    fake_A = 0.5*G_BA(real_B).data + 1.0

    # Determine approximate time left
    iter_done = len(test_dataloader) + i
    batches_left = len(test_dataloader) - iter_done

    # If at sample interval save image
    if iter_done % opt.sample_interval == 0:
        sample_images(iter_done)