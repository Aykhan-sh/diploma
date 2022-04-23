import torch
from tqdm import tqdm
from defs import STAGES, IDX2CHAR
from torch.utils.data import DataLoader
from torch import nn
from typing import Dict, Type
from utils import greedy_decode, get_ground_truth
from jiwer import wer, cer


def train(
    dataloaders: Dict[STAGES, DataLoader],
    model: Type[torch.nn.Module],
    optimizer,
    loss_fn,
    epochs,
    device: str,
):
    total_loss = 0
    wers = 0
    train_history = {"loss": [], "wer": []}
    val_history = {"loss": [], "wer": [], "cer": []}
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        wers = 0
        model.train()
        for step, (X, y, input_percentages, target_sizes) in enumerate(
            tqdm(dataloaders["train"])
        ):
            optimizer.zero_grad()
            input_sizes = input_percentages.mul_(int(X.size(3))).int()
            X, y = X.to(device), y.to(device)
            preds, output_sizes = model(X, input_sizes)
            log_probs = nn.functional.log_softmax(preds, dim=-1)
            loss = loss_fn(log_probs, y, output_sizes, target_sizes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            ground_truth = get_ground_truth(y.cpu().detach(), target_sizes)
            decoded = greedy_decode(
                nn.functional.softmax(preds.cpu().detach(), dim=-1).transpose(1, 0),
                IDX2CHAR,
            )
            wers += wer(ground_truth, decoded)
            if step % 1000 == 0:
                print(f"WER: {round(wers/(step+1), 3)}")
                print(f"Decoded: {decoded[-1]}")
                print(f"Ground Truth: {ground_truth[-1]}")
        train_history["loss"].append(total_loss / len(dataloaders["train"]))
        train_history["wer"].append(wers / len(dataloaders["train"]))

        model.eval()
        total_loss = 0
        wers = 0
        cers = 0
        with torch.no_grad():
            for X, y, input_percentages, target_sizes in tqdm(dataloaders["val"]):
                input_sizes = input_percentages.mul_(int(X.size(3))).int()
                X, y = X.to(device), y.to(device)
                preds, output_sizes = model(X, input_sizes)
                log_probs = nn.functional.log_softmax(preds, dim=-1)
                probs = nn.functional.softmax(preds, dim=-1).cpu().detach()
                loss = loss_fn(log_probs, y, output_sizes, target_sizes)
                total_loss += loss.item()
                decoded = greedy_decode(probs.transpose(0, 1), IDX2CHAR)
                ground_truth = get_ground_truth(y.cpu().detach(), target_sizes)
                wers += wer(ground_truth, decoded)
                cers = cer(ground_truth, decoded)

            val_history["loss"].append(total_loss / len(dataloaders["val"]))
            val_history["wer"].append(wers / len(dataloaders["val"]))
            val_history["cer"].append(cers / len(dataloaders["val"]))

        print(f"Epoch: {epoch+1}")
        print(
            f"Train Loss: {train_history['loss'][-1]}, Train Wer: {train_history['wer'][-1]}"
        )
        print(
            f"Val Loss: {val_history['loss'][-1]}, Val Wer: {val_history['wer'][-1]}, Val Cer: {val_history['cer'][-1]}"
        )
        print("Decoded:", decoded[:3])
        print("Ground Truth:", ground_truth[:3])
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            "model_epoch{}_wer{}.pth".format(epoch, round(val_history["wer"][-1], 3)),
        )

