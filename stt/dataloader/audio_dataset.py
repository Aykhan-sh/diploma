import torchaudio
from torch.utils.data import Dataset
import torch


class AudioDataset(Dataset):
    def __init__(
        self,
        data_frame,
        transformation,
        chr2idx,
        device,
        sample_rate=16000,
        audio_augm=None,
        spec_augm=None,
    ):
        self.audio_paths = data_frame.audio_path.to_list()
        self.chr2idx = chr2idx
        self.text = data_frame.text.apply(lambda x: self.text_preprocess(x))
        self.device = device
        self.transformation = transformation
        self.spec_augm = spec_augm
        self.sample_rate = sample_rate
        self.audio_augm = audio_augm

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        text = self.text[idx]
        signal, sr = torchaudio.load(audio_path)
        spect = self.audio_preprocess(signal, sr)
        spec = self.apply_spec_augm(spect)
        # transcript = self.text_preprocess(text)

        return spect, text

    def audio_preprocess(self, signal, sr):
        if sr != self.sample_rate:
            signal = self.resample(signal, sr)

        signal = self.apply_audio_augm(signal)
        signal = self.to_mono(signal)
        spect = self.transformation(signal)
        spect = torch.log1p(spect)

        return spect

    def resample(self, signal, source_sr):
        resampler = torchaudio.transforms.Resample(
            orig_freq=source_sr, new_freq=self.sample_rate
        )
        signal = resampler(signal)

        return signal

    def apply_audio_augm(self, signal):
        if self.audio_augm is None:
            return signal

        signal = signal.cpu()
        for augm in self.audio_augm:
            signal = augm(signal)

        return signal

    def to_mono(self, signal):
        if signal.shape[0] == 0:
            signal = signal.squeeze()
        else:
            signal = signal.mean(axis=0)

        return signal

    def apply_spec_augm(self, spect):
        if self.spec_augm is None:
            return spect

        spect = spect.unsqueeze(0)
        for augm in self.spec_augm:
            spect = augm(spect)

        return spect

    def text_preprocess(self, text):
        transcript = list(filter(None, [self.chr2idx.get(x) for x in list(text)]))
        return transcript

    def batch_preprocessing(batch):
        batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
        longest_sample = batch[0][0]
        freq_size = longest_sample.size(0)
        minibatch_size = len(batch)
        max_seqlength = longest_sample.size(1)
        inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
        input_percentages = torch.FloatTensor(minibatch_size)
        target_sizes = torch.IntTensor(minibatch_size)
        targets = []
        for x in range(minibatch_size):
            sample = batch[x]
            tensor = sample[0]
            target = sample[1]
            seq_length = tensor.size(1)
            inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
            input_percentages[x] = seq_length / float(max_seqlength)
            target_sizes[x] = len(target)
            targets.extend(target)
        targets = torch.tensor(targets, dtype=torch.long)

        return inputs, targets, input_percentages, target_sizes


def collate_fn(batch):
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = batch[0][0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.tensor(targets, dtype=torch.long)

    return inputs, targets, input_percentages, target_sizes
