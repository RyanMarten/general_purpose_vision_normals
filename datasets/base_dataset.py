
from os import makedirs
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as NF

from pathlib import Path
from torch.utils.data import Dataset

from PIL import Image
import numpy as np
from pathlib import Path
import cv2

class BaseDataset(Dataset):
    def __init__(self, root, 
                 mode = 'train') -> Dataset:
        """Base class for the datasets.

        We assume that the file is structured as:
        ```
        images/
            00000000.png
        depths/ 
            00000000.pfm
        normals/
            00000000.png
        ```
        :param root: path to the root folder of the dataset. 
               mode: the training or validation mode
        :returns: Instance of the PyTorch Dataset class
        """
        super().__init__()
        self.root = Path(root) 
        self.mode = mode

        self.images_files = self.make_dataset(self.root,'images')
        self.normals_files = self.make_dataset(self.root,'normals')

    def make_dataset(self, dir, data_type):
        data = []
        dir = Path(dir)
        dir = dir / data_type
        for data_path in sorted(dir.iterdir()):
            if (not data_path.name[-3:] == 'jpg') & (not data_path.name[-3:] == 'png') &\
                (not data_path.name[-3:] == 'pfm') & (not data_path.name[-3:] == 'npy') : 
                continue
            data.append(data_path)
        return data

    def load_image (self, image_path):
        image = F.to_tensor(Image.open(image_path).convert('RGB')).float()
        return image

    def load_normals (self, normals_path):
        normals = F.to_tensor(Image.open(normals_path)).float() * 2 - 1
        return normals            

    def data_preprocessing(self, image, normals):
        TH, TW = 480, 640
        H, W = image.shape[-2:]
        if H > TH:
            crop_function = transforms.RandomCrop((TH, TW))
            input_stack = torch.cat((image, normals), dim = 0)
            input_stack = crop_function(input_stack)
            image, normals = input_stack[:3], input_stack[3:]
        return image, normals

    def __len__(self) -> int:
        return len(self.images_files)

    def __getitem__(self, index) -> tuple:
        """Returns sample from the loadable

        :param index: index of the sample in the samples list

        :returns: tuple of loadables, such as images, cameras and depths
        """
        single_batch = {}
        image = self.load_image(self.images_files[index])
        normals = self.load_normals(self.normals_files[index])
        
        if self.mode == 'train':
            image, normals = self.data_preprocessing(image, normals)
        elif 'sunrgbd' in str(self.root): 
            # Crop frames of SUNRGBD so that its resolution is dividable by 16
            image, normals = self.cropping(image, normals)

        single_batch['image'] = image
        single_batch['normals'] = normals

        return single_batch
