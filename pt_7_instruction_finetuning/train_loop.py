import torch
from torch import nn
from torch.utils.data import DataLoader

import torchmetrics
from torchmetrics.classification.accuracy import MulticlassAccuracy

import tiktoken
from pt_5_pretraining_on_unlabeled_data.training_model.model import GPT2Model

def test_model(model: GPT2Model,
          loss_fn: nn.CrossEntropyLoss,
          test_loader: DataLoader,
          device: torch.device):

    test_loss = 0
    model.eval()
    with torch.inference_mode():

        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            logits = model(X)

            loss = loss_fn(logits.view(logits.shape[0] * logits.shape[1], logits.shape[2]), y.flatten())
            preds = torch.argmax(logits, dim=-1)

            test_loss += loss.item()

        test_loss /= len(test_loader)

    return test_loss

def generate_text_test(max_new_tokens, sample_text, model, device, temp=0.0, top_k=None, eos_id=None):
    model.to(device)
    tokenizer = tiktoken.get_encoding("gpt2")
    ids = tokenizer.encode(sample_text, allowed_special={"<|endoftext|>"})
    ids = torch.tensor(ids).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():

        for _ in range(max_new_tokens):
            logits = model(ids)
            logits = logits[:, -1].squeeze()

            if top_k:
                top_logits, top_pos = torch.topk(logits, top_k)
                logits = torch.where(logits >= top_logits[-1], logits, torch.tensor(-torch.inf, device=device))

            if temp > 0.0:
                probs = torch.softmax(logits / temp, dim=-1)
                id_next = torch.multinomial(probs, 1)

            else:
                id_next = torch.argmax(logits, dim=-1, keepdim=True)


            if id_next == eos_id:
                break
            ids = torch.cat((ids, id_next.unsqueeze(0)), dim=1)

    ids = ids.squeeze().tolist()
    text = tokenizer.decode(ids)

    return text

def train(model: GPT2Model,
          loss_fn: nn.CrossEntropyLoss,
          optimizer: torch.optim.AdamW,
          scheduler: torch.optim.lr_scheduler.StepLR,
          train_loader: DataLoader,
          val_loader: DataLoader,
          test_loader: DataLoader,
          device: torch.device,
          EPOCHS: int):
    train_losses, val_losses, epochs = [], [], []

    for epoch in range(1, EPOCHS + 1):

        train_loss = 0
        model.train()

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            logits = model(X)

            loss = loss_fn(logits.view(logits.shape[0] * logits.shape[1], logits.shape[2]), y.flatten())
            preds = torch.argmax(logits, dim=-1)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        val_loss = 0
        model.eval()

        with torch.inference_mode():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)

                logits = model(X)

                loss = loss_fn(logits.view(logits.shape[0] * logits.shape[1], logits.shape[2]), y.flatten())
                preds = torch.argmax(logits, dim=-1)

                val_loss += loss.item()

            val_loss /= len(val_loader)
        val_losses.append(val_loss)
        epochs.append(epoch)

        scheduler.step()

        print(f"\nEpoch: {epoch} | Train loss: {train_loss}")
        print(f"          Val loss: {val_loss}")

    test_loss = test_model(model, loss_fn, test_loader, device)
    print(f"          Final results: Test loss: {test_loss}")

    return model, train_losses, val_losses, epochs