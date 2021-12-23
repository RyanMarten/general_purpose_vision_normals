from datasets.base_dataset import BaseDataset
from datasets.base_datamodule import BaseDataModule
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as NF
from PIL import Image
import numpy as np
import torch
import math

class DTUDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(DTUDataset, self).__init__(*args, **kwargs)

    def load_depth(self, depth_path):
        depth = torch.tensor(np.load(depth_path))
        return depth
        


class DTU(BaseDataModule):
    def __init__(self, root, mode, *args, **kwargs):
        super(DTU, self).__init__(root, mode, *args, **kwargs)

    def get_dataset_dirname(self):
        return 'dtu'

    def get_set_ids(self) -> list:
        dirname = self.root / self.get_dataset_dirname()
        if(self.mode == 'train'):
            raise RuntimeError('Doesn\'t provide training sets for DTU')
        else:
            return [f'val/{l.name}' for l in sorted((dirname / 'val').iterdir()) if l.is_dir()]

    def get_dataset_class(self)->BaseDataset:
        return DTUDataset
