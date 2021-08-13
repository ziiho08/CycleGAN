import glob
import os
import PIL
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from random import sample
import numpy as np
import natsort

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self, root,folder_name = 'mask',folder_num=1):

        self.files_A = sorted(glob.glob(os.path.join(root, f'domainX/{folder_name}/{folder_num}') + "/*.*"))
        self.files_A_sort = natsort.natsorted(self.files_A)
        self.files_B = sorted(glob.glob(os.path.join(root, f'domainY') + "/*.*"))
        self.shuffle_B = sample(self.files_B, len(self.files_B))[:3500]


    def __getitem__(self, index):

        item_A = self.transform_A(Image.open(self.files_A_sort[index % len(self.files_A_sort)]).convert('L'))
        #print(f'item_A.shape : {item_A.shape}',len(self.files_A))

        item_B = self.transform_B(Image.open(self.shuffle_B[index % len(self.shuffle_B)]).convert('L'))
        #print(f'item_B.shape : {item_B.shape}')

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A_sort), len(self.shuffle_B))

    def transform_A(self, image):
        transform_ = transforms.Compose([transforms.ToTensor()])
        return transform_(image)

    def transform_B(self, image):
        transform_ = transforms.Compose([transforms.ToTensor(),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.RandomRotation(90)])

        return transform_(image)