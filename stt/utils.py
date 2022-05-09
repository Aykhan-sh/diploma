from typing import Literal, Dict
import pandas as pd

from glob import glob
from tqdm import tqdm
import os

import torch
import torchaudio
from torch.utils.data import DataLoader
from dataloader.audio_dataset import AudioDataset, collate_fn
from defs import CHAR2IDX, STAGES, IDX2CHAR


def create_df(data_path: str):
    data_path = 'data'
    audio_paths = glob(os.path.join(data_path, 'Audios', '*.wav'))
    text = []
    for audio_path in tqdm(audio_paths):
        text_name = os.path.basename(audio_path).replace('.wav', '.txt')
        text.append(
            open(
                os.path.join(data_path, 'Transcriptions', text_name),
                encoding="utf8"
            ).read()
        )
    df = pd.DataFrame({'audio_path': audio_paths, 'text': text})
    return df



def get_dataloaders(
    dataframes: Dict[STAGES, pd.DataFrame],
    dataloader_config: Dict[STAGES, dict],
    transformation: torchaudio.transforms.Spectrogram,
    char2idx: Dict[str, int] = CHAR2IDX,
    device: str = 'cuda'
) -> Dict[STAGES, DataLoader]:
    assert set(dataframes.keys()) == set(
        dataloader_config.keys()
    ), "keys of dataframes and dataloader_config must be the same"
    result = {}
    for stage in dataframes.keys():
        dataset = AudioDataset(dataframes[stage], transformation, char2idx, device)
        dataloader_config[stage]["shuffle"] = True
        if stage in ["val", "test"]:
            dataloader_config[stage]["shuffle"] = False
        result[stage] = DataLoader(
            dataset, **dataloader_config[stage], collate_fn=collate_fn
        )
    return result


def get_ground_truth(y, target_sizes, idx2char=IDX2CHAR):
    texts = []
    idx = 0
    for size in target_sizes:
        text = ""
        for i in range(size.item()):
            text += IDX2CHAR[y[idx + i].item()]

        texts.append(text)
        idx += size

    return texts


def greedy_decode(probas, idx2char=IDX2CHAR, blank_idx=0):
    max_values, classes = torch.max(probas, dim=-1)
    texts = []
    for sequence in range(len(classes)):
        sequence_len = len(classes[sequence])
        text = ""
        for i in range(sequence_len):
            char = idx2char[classes[sequence][i].item()]
            if char != idx2char[blank_idx]:
                if i != 0 and char == idx2char[classes[sequence][i - 1].item()]:
                    continue
                text += char
        texts.append(text)

    return texts

