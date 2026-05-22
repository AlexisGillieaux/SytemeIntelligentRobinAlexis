import os
import torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET"))
print("CUDA dispo:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)