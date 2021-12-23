from .base_datamodule import BaseDataModule
from datasets.base_dataset import BaseDataset
from utils.io import read_pfm
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as NF
import cv2
import torch
import numpy as np

class SUNRGBDDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(SUNRGBDDataset, self).__init__(*args, **kwargs)

    def load_depth(self, depth_path):
        depth = cv2.imread(str(depth_path), (cv2.IMREAD_ANYDEPTH))
        depth = (np.right_shift(depth, 3) | np.right_shift(depth, -13)) / 1000.
        return torch.tensor(depth).float()

    def cropping(self, image, normals):
        TH, TW = 512, 720
        H, W = image.shape[-2:]
        if H > TH:
            crop_function = transforms.CenterCrop((TH, TW))
            input_stack = torch.cat((image, normals), dim = 0)
            input_stack = crop_function(input_stack)
            image, normals = input_stack[:3], input_stack[3:]
        return image, normals        


class SUNRGBD(BaseDataModule):
    def __init__(self, root, mode, *args, **kwargs):
        super(SUNRGBD, self).__init__(root, mode, *args, **kwargs)
        self.datasets = ['sun3d', 
                         'sunrgbd_ori']

    def get_dataset_dirname(self):
        return 'sunrgbd'

    def get_set_ids(self) -> list:
        dirname = self.root / self.get_dataset_dirname()
        if(self.mode == 'train'):
            return [f'{self.datasets[0]}/train/{l.name}' for l in sorted((dirname / self.datasets[0] / 'train').iterdir()) if l.is_dir()] + \
            [f'{self.datasets[1]}/train/{l.name}' for l in sorted((dirname / self.datasets[1] / 'train').iterdir()) if l.is_dir()]
        else:
            return [f'{self.datasets[0]}/train/{l.name}' for l in sorted((dirname / self.datasets[0] / 'train').iterdir()) if l.is_dir()] + \
            [f'{self.datasets[1]}/train/{l.name}' for l in sorted((dirname / self.datasets[1] / 'train').iterdir()) if l.is_dir()]
    def get_dataset_class(self)->BaseDataset:
        return SUNRGBDDataset
