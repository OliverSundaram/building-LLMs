import torch

# TASK: Determine the context vector for input 6

inputs = torch.tensor([
    [0.42, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55]
])

qkv_shape = [3, 2]

W_query = torch.rand(size=qkv_shape)
W_keys = torch.rand(size=qkv_shape)
W_values = torch.rand(size=qkv_shape)

q_6 = inputs[5] @ W_query

keys = inputs @ W_keys
values = inputs @ W_values

atten_weights_6 = q_6 @ keys.T
atten_weights_6 = torch.softmax(atten_weights_6 / qkv_shape[1]**0.5, dim=-1)

context_vec_6 = atten_weights_6 @ values
print(context_vec_6)