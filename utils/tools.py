import torch

def calculate_accuracy(model, dataloader):
    running_acc = 0.0
    total_size = 0
    for i, (inputs, labels) in enumerate(dataloader):
        outputs = model(inputs)
        _, y_pred = torch.max(outputs.data, 1)
        running_acc += (y_pred == labels).sum().item()
        total_size += len(inputs)

    return running_acc / total_size