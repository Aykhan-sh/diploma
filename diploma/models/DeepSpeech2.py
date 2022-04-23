from torch import nn
import math
from ..defs import TARGET_SR


class DeepSpeech2(nn.Module):
    def __init__(
        self,
        n_feature: int,
        n_hidden: int,
        n_class: int,
        window_size: int,
        dropout: float = 0,
        max_clip_relu: float = 20,
        n_rnn_layer: int = 3,
        target_sr: int = TARGET_SR,
    ):
        super(DeepSpeech2, self).__init__()
        self.n_hidden = n_hidden
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
        )

        rnn_input_size = int(math.floor((target_sr * window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32
        self.bi_rnn = nn.GRU(
            rnn_input_size, n_hidden, bidirectional=True, num_layers=n_rnn_layer
        )
        self.out = nn.Sequential(nn.Linear(n_hidden, n_class, bias=False))

    def forward(self, x, input_sizes):
        output_sizes = self.get_output_lenght(input_sizes)
        x = self.conv_block(x)
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))
        x = x.permute(2, 0, 1)
        x, _ = self.bi_rnn(x)
        x = x[:, :, : self.n_hidden] + x[:, :, self.n_hidden :]
        t, n, h = x.size(0), x.size(1), x.size(2)
        x = x.view(t * n, -1)
        x = self.out(x)
        x = x.view(t, n, -1)

        return x, output_sizes

    def get_output_lenght(self, input_lenght):
        seq_len = input_lenght
        for block in self.conv_block.modules():
            if type(block) == nn.modules.conv.Conv2d:
                seq_len = (
                    seq_len
                    + 2 * block.padding[1]
                    - block.dilation[1] * (block.kernel_size[1] - 1)
                    - 1
                ) // block.stride[1] + 1
        return seq_len.int()
