from torch import nn
from jiwer import wer, cer

class DeepSpeech(nn.Module):
    def __init__(
        self,
        n_feature: int,
        n_hidden: int,
        n_class: int,
        dropout: float = 0,
        max_clip_relu: float = 20,
    ):
        super(DeepSpeech, self).__init__()
        self.n_hidden = n_hidden
        self.fc_block = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            nn.Hardtanh(0, max_clip_relu),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.Hardtanh(0, max_clip_relu),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.Hardtanh(0, max_clip_relu),
            nn.Dropout(dropout),
        )
        self.bi_rnn = nn.GRU(n_hidden, n_hidden, bidirectional=True, num_layers=1)
        self.out = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Hardtanh(0, max_clip_relu),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_class),
        )

    def forward(self, x, input_sizes):
        x = x.permute(0, 1, 3, 2)
        output_sizes = input_sizes
        x = self.fc_block(x)
        x = x.squeeze(1)
        x = x.transpose(0, 1)
        x, _ = self.bi_rnn(x)
        x = x[:, :, : self.n_hidden] + x[:, :, self.n_hidden :]
        x = self.out(x)

        return x, output_sizes
