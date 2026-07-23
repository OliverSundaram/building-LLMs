import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

with open("the-verdict.txt", "r") as file:
    raw_text = file.read()

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def gen_dataloader(txt, tokenizer, batch_size=8, max_length=64, stride=64, shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDataset(txt, tokenizer=tokenizer, max_length=max_length, stride=stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

tokenizer = tiktoken.get_encoding("gpt2")
dataloader = gen_dataloader(raw_text, tokenizer, batch_size=8, max_length=4, stride=4, shuffle=False)


# Make token embedding layer
vocab_size = tokenizer.n_vocab
output_dim = 256

torch.manual_seed(42)
torch.cuda.manual_seed(42)
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

inputs, targets = next(iter(dataloader))

token_embedding = token_embedding_layer(inputs)


# Make position embedding layer
vocab_size = 4
output_dim = 256

torch.manual_seed(42)
torch.cuda.manual_seed(42)
pos_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

pos_embedding = pos_embedding_layer(torch.arange(4))

input_embedding = token_embedding + pos_embedding