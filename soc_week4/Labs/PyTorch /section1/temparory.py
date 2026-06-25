import torch

print(torch.__version__)
print(torch.backends.mps.is_available())  # should print True
print(torch.backends.mps.is_built())      # should print True