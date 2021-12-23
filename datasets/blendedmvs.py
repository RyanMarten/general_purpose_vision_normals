from datasets.base_dataset import BaseDataset
from datasets.base_datamodule import BaseDataModule
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as NF
from PIL import Image
import numpy as np
import torch
import math
from utils.io import read_pfm


class BlendedMVSDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(BlendedMVSDataset, self).__init__(*args, **kwargs)

    def load_depth(self, depth_path):
        depth = torch.tensor(read_pfm(depth_path))
        return depth
        


class BlendedMVS(BaseDataModule):
    def __init__(self, root, mode, *args, **kwargs):
        super(BlendedMVS, self).__init__(root, mode, *args, **kwargs)

    def get_dataset_dirname(self):
        return 'Blended_MVS'

    def get_set_ids(self) -> list:
        dirname = self.root / self.get_dataset_dirname()
        if(self.mode == 'train'):
            raise [f'train/{l.name}' for l in sorted((dirname / 'train').iterdir()) if l.is_dir()]
        else:
            return [f'val/{l.name}' for l in sorted((dirname / 'val').iterdir()) if l.is_dir()]

    def get_dataset_class(self)->BaseDataset:
        return BlendedMVSDataset
