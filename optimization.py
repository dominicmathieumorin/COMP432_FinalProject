import torch

model = torch.jit.load('models/latest_model.pt')
model.eval()
