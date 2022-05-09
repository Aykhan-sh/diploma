from email.policy import default
import torch
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset

from typing import Dict, List


class TextDataset(Dataset):
    def __init__(self, text: List[str], stoi: Dict[str, int], seq_len):
        text_clipping_len = (len(text) // seq_len) * seq_len
        self.text = text[:text_clipping_len]
        self.stoi = stoi
        self.seq_len = seq_len

    def __getitem__(self, idx: int):
        text = self.text[idx : idx + self.seq_len + 1]
        input_tensor = torch.tensor([self.stoi.get(char, self.stoi['ukn']) for char in text[:-1]])
        target_tensor = torch.tensor([self.stoi.get(char, self.stoi['ukn']) for char in list(text[1:])])
        return input_tensor, target_tensor

    def __len__(self):
        return len(self.text) - self.seq_len


def merge(sequences):
    lengths = [len(seq) for seq in sequences]
    padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = seq[:end]
    return padded_seqs


def collate_fn(batch):
    input_tensors = [x[0] for x in batch]
    input_tensors = merge(input_tensors)

    target_tensors = [x[1] for x in batch]
    target_tensors = merge(target_tensors)
    return input_tensors, target_tensors

