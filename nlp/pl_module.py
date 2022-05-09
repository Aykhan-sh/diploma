from turtle import forward
from numpy import isin
from pytorch_lightning import LightningModule
import torch


class TextModule(LightningModule):
    def __init__(self, model, stoi, itos, criterion, lr, device="cuda") -> None:
        super().__init__()
        self.model = model
        self.stoi = stoi
        self.itos = itos
        self.criterion = criterion
        self.lr = lr

    def forward(self, x, hidden_states):
        return self.model(x, hidden_states)

    def training_step(self, batch, batch_idx):
        x, y = batch
        hidden_states = self.init_state(x.shape[0])
        y_pred, (self.state_h, self.state_c) = self(x, hidden_states)
        loss = self.criterion(y_pred.transpose(1, 2), y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return [optimizer]

    def init_state(self, batch_size, device=None):
        if device is None:
            device = self.device
        return (
            torch.zeros(self.model.num_layers, batch_size, self.model.lstm_size).to(
                device
            ),
            torch.zeros(self.model.num_layers, batch_size, self.model.lstm_size).to(
                device
            ),
        )

    @torch.no_grad()
    def generate(self, text: str, max_length: int):
        self.model.eval()
        if isinstance(text, str):
            text = text.split(" ")
        input_tensor = torch.tensor(
            [self.stoi.get(char, self.stoi["ukn"]) for char in text]
        )[None, :]
        input_tensor = input_tensor.to(self.device).long()
        hidden_states = self.init_state(1)
        result = []
        for i in range(max_length):
            input_tensor, hidden_states = self.model(input_tensor, hidden_states)
            topv, topi = input_tensor.topk(1)
            topi = topi[0][0]
            result.append(topi)
            input_tensor = torch.tensor([[topi]], device=self.device)
        result = [self.itos[i] for i in result]
        return result
