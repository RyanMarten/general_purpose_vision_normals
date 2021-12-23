from pathlib import Path
from tqdm import tqdm
from torch.utils.data.dataset import ConcatDataset

class BaseDataModule:
    def __init__(self, root, mode, *args, **kwargs):
        self.root = Path(root)
        self.mode = mode
        self.args = args
        self.kwargs = kwargs
        self.datasets = []

    def get_set_ids(self)->list:
        raise NotImplementedError

    def get_dataset_dirname(self):
        raise NotImplementedError

    def get_dataset_class(self):
        raise NotImplementedError

    def concatenate_dataset(self):
        # Concatenate dataset
        set_ids = self.get_set_ids()
        data_root = Path(self.root) / self.get_dataset_dirname()
        Dataset = self.get_dataset_class()
        args = self.args
        kwargs = self.kwargs
        datasets = []
        for set_id in tqdm(set_ids, desc="Concatenating Datasets..."):
            dataset = Dataset(data_root / set_id, mode = self.mode, *args, **kwargs)
            datasets.append(dataset)
        self.datasets = ConcatDataset(datasets)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, i):
        return self.datasets[i]
