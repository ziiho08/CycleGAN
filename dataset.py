import glob
import os
import PIL
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from random import sample

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self, root,folder_name = 'defualt_mask',folder_num=12):

        self.files_A = sorted(glob.glob(os.path.join(root, f'domainX/{folder_name}/{folder_num}') + "/*.*"))
        self.shuffle_A = sample(self.files_A, len(self.files_A))[:2000]
        self.files_B = sorted(glob.glob(os.path.join(root, f'domainY') + "/*.*"))

    def __getitem__(self, index):

        item_A = self.transform_A(Image.open(self.shuffle_A[index % len(self.shuffle_A)])
                                  .resize((128, 128), PIL.Image.BICUBIC).convert('L'))
        #print(f'item_A.shape : {item_A.shape}')

        item_B = self.transform_B(Image.open(self.files_B[index % len(self.files_B)]).convert('L'))
        #print(f'item_B.shape : {item_B.shape}')

        return {'A': item_A*10, 'B': item_B}

    def __len__(self):
        # print(len(self.files_A), len(self.files_B))
        return max(len(self.shuffle_A), len(self.files_B))

    def transform_A(self, image):
        transform_ = transforms.Compose([transforms.ToTensor()])
        return transform_(image)

    def transform_B(self, image):
        transform_ = transforms.Compose([transforms.ToTensor(),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.RandomRotation(90)])

        return transform_(image)
