from flash_attn import flash_attn_func

import torch

x = torch.randn(1, 10, 10, 10, device="cuda", dtype=torch.bfloat16)
y = flash_attn_func(x, x, x)
print("flash_attn_func output shape:", y.shape)