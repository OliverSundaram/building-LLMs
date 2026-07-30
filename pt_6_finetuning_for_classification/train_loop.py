import torch
from torch import nn
from torch.utils.data import DataLoader

import torchmetrics
from torchmetrics.classification.accuracy import MulticlassAccuracy

import tiktoken
from pt_5_pretraining_on_unlabeled_data.training_model.model import GPT2Model

def test_model(model: GPT2Model,
               sample_text: str,
               tokenizer,
               device: torch.device):
    vocab = {0: "ham",
             1: "spam"}

    ids = tokenizer.encode(sample_text, allowed_special={"<|endoftext|>"})
    ids = torch.tensor(ids).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():

        logits = model(ids)

        last_logits = logits[:, -1]
        pred = int(torch.argmax(last_logits, dim=-1))

    return vocab[pred]



def train(model: GPT2Model,
          loss_fn: nn.CrossEntropyLoss,
          acc_fn: MulticlassAccuracy,
          optimizer: torch.optim.AdamW,
          scheduler: torch.optim.lr_scheduler.StepLR,
          train_loader: DataLoader,
          val_loader: DataLoader,
          device: torch.device,
          EPOCHS: int,
          tokenizer,
          sample_text: str):
    train_losses, val_losses, train_accs, val_accs, epochs = [], [], [], [], []

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = 0, 0
        model.train()

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            last_logits = logits[:, -1]

            loss = loss_fn(last_logits, y)
            preds = torch.argmax(last_logits, dim=-1)
            acc = acc_fn(preds, y)

            train_loss += loss.item()
            train_acc += acc.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = 0, 0
        model.eval()

        with torch.inference_mode():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)

                logits = model(X)
                last_logits = logits[:, -1]

                loss = loss_fn(last_logits, y)
                preds = torch.argmax(last_logits, dim=-1)
                acc = acc_fn(preds, y)

                val_loss += loss.item()
                val_acc += acc.item()

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        epochs.append(epoch)

        scheduler.step()

        print(f"\nEpoch: {epoch} | Train loss: {train_loss} | Train acc: {train_acc}")
        print(f"          Val loss: {val_loss} | Val acc: {val_acc}")

        text = test_model(model, sample_text, tokenizer, device)
        print(f"Model prediction on: {sample_text}\n{text}")

    return model, train_losses, train_accs, val_losses, val_accs, epochs