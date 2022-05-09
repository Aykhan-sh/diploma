from pytorch_lightning import LightningDataModule
from .dataset import TextDataset, collate_fn
from typing import Dict, List
from torch.utils.data import DataLoader


class TextDatamodule(LightningDataModule):
    def __init__(
        self,
        text: List[str],
        stoi: Dict[str, int],
        seq_len: int,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.text = text
        self.stoi = stoi
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_set = TextDataset(self.text, self.stoi, self.seq_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

