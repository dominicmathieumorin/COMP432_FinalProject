import torch
import numpy as np
from torch.utils.data import DataLoader

from model import Model
from utils.dataset import MappilaryDataset
from torch.utils.tensorboard import SummaryWriter

from utils.tools import calculate_accuracy
from utils.vizualization import log_random_images, log_class_distribution

writer = SummaryWriter()

# pseudo randomness
np.random.seed(0)
torch.manual_seed(0)

# Step 0: Setup dataset
batch_size = 100
train_dataset = MappilaryDataset('data/train.pkl')
test_dataset = MappilaryDataset('data/test.pkl')
val_dataset = MappilaryDataset('data/val.pkl')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# log class distribution
log_class_distribution("train_class_histogram", writer, train_dataset)
log_class_distribution("test_class_histogram", writer, test_dataset)
log_class_distribution("val_class_histogram", writer, val_dataset)

# log some images
log_random_images('train_images', writer, train_dataset)
log_random_images('test_images', writer, test_dataset)
log_random_images('val_images', writer, val_dataset)

# Step 1: create model
model = Model()

# Step 2: define loss & optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Step 3: run the training loop
num_epoch = 10

for epoch in range(1, num_epoch + 1):
    model.train()
    running_train_acc = 0.0
    for i, (inputs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()  # Zero your gradients for every batch!
        outputs = model(inputs)  # Make predictions for this batch
        loss = loss_fn(outputs, labels)  # Compute the loss and its gradients
        loss.backward()
        optimizer.step()

        # calculate train accuracy
        _, y_pred = torch.max(outputs.data, 1)
        batch_train_accuracy = (y_pred == labels).sum().item()
        running_train_acc += batch_train_accuracy

        if i % 100 == 0 and i > 0:
            print(f"Epoch: {epoch} - Minibatch #{i} Accuracy: {batch_train_accuracy / batch_size:.2%}")

    train_accuracy = running_train_acc / len(train_dataset)

    # calculate accuracy
    model.eval()
    test_accuracy = calculate_accuracy(model, test_dataloader)
    val_accuracy = calculate_accuracy(model, val_dataloader)

    print(f"Finished processing epoch: {epoch}")
    print(f"Train Accuracy: {train_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Val Accuracy: {val_accuracy:.2%}")

    writer.add_scalars('Accuracy', {
        'train': train_accuracy,
        'test': test_accuracy,
        'val': val_accuracy}, epoch)

model_scripted = torch.jit.script(model)  # Export to TorchScript
model_scripted.save('models/latest_model.pt')  # Save

writer.close()
