from datasets.base_dataset import BaseDataset
from datasets.base_datamodule import BaseDataModule
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as NF
from PIL import Image
import numpy as np
import torch
import math

class NYUv2Dataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(NYUv2Dataset, self).__init__(*args, **kwargs)

    def load_depth(self, depth_path):
        depth = torch.tensor(np.load(depth_path))
        return depth
        


class NYUv2(BaseDataModule):
    def __init__(self, root, mode, *args, **kwargs):
        super(NYUv2, self).__init__(root, mode, *args, **kwargs)

    def get_dataset_dirname(self):
        return 'nyuv2'

    def get_set_ids(self) -> list:
        dirname = self.root / self.get_dataset_dirname()
        if(self.mode == 'train'):
            return ['train']
        else:
            return ['val']

    def get_dataset_class(self)->BaseDataset:
        return NYUv2Dataset
