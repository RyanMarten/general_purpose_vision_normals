from datasets.base_dataset import BaseDataset
from datasets.base_datamodule import BaseDataModule
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as NF
import torch
import math

class ScanNetDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(ScanNetDataset, self).__init__(*args, **kwargs)

    def load_depth(self, depth_path):
        raise NotImplementedError('Depth files are not provided')
        

class ScanNet(BaseDataModule):
    def __init__(self, root, mode, *args, **kwargs):
        super(ScanNet, self).__init__(root, mode, *args, **kwargs)

    def get_dataset_dirname(self):
        return 'scannet'

    def get_set_ids(self) -> list:
        dirname = self.root / self.get_dataset_dirname()
        if(self.mode == 'train'):
            return [f'train/{l.name}' for l in sorted((dirname / 'train').iterdir()) if l.is_dir()]
        else:
            return [f'val/{l.name}' for l in sorted((dirname / 'val').iterdir()) if l.is_dir()]

    def get_dataset_class(self)->BaseDataset:
        return ScanNetDataset
