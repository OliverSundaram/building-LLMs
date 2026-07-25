import torch

# TASK: Determine the context vector for all inputs at the same time

inputs = torch.tensor([
    [0.42, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55]
])

W_query = torch.nn.Linear(3, 2, bias=False)
W_key = torch.nn.Linear(3, 2, bias=False)
W_value = torch.nn.Linear(3, 2, bias=False)

queries = W_query(inputs)
keys = W_key(inputs)
values = W_value(inputs)

atten_score = queries @ keys.T
atten_weights = torch.softmax(atten_score / keys.shape[1]**0.5, dim=1)
context_vec = atten_weights @ values

print(context_vec)