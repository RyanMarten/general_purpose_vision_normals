from datasets.base_dataset import BaseDataset
from datasets.base_datamodule import BaseDataModule
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as NF
from PIL import Image
import numpy as np
import torch
import math

class KITTIDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

    def load_depth(self, depth_path):
        depth = np.array(Image.open(depth_path), dtype=np.uint16)
        depth = torch.tensor(depth.astype(np.float32) / 256.)
        return depth
        

class KITTI(BaseDataModule):
    def __init__(self, root, mode, *args, **kwargs):
        super(KITTI, self).__init__(root, mode, *args, **kwargs)

    def get_dataset_dirname(self):
        return 'kitti'

    def get_set_ids(self) -> list:
        dirname = self.root / self.get_dataset_dirname()
        if(self.mode == 'train'):
            raise RuntimeError('Doesn\'t provide training sets for DTU')
        else:
            return ['val']

    def get_dataset_class(self)->BaseDataset:
        return KITTIDataset
