import re

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.nn import CrossEntropyLoss
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os

from dataloader import fused_CustomDataset

from  FCN import FCN

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

l = torch.load('./normed_combined.pt')
second_features = l['data']
second_labels = l['label']


train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
])

data_path = './cwt/'
second_features = second_features

image_labels = []
for file in sorted(os.listdir(data_path)):
    folder_path = os.path.join(data_path, file)
    if os.path.isdir(folder_path):
        label = int(file) - 1
        label_files = sorted(os.listdir(folder_path))
        image_labels += [(label, image_file) for image_file in label_files]
image_paths = []
for file in sorted(os.listdir(data_path)):
    folder_path = os.path.join(data_path, file)
    if os.path.isdir(folder_path):
        # image_files = sorted(os.listdir(folder_path))
        image_files = sorted(os.listdir(folder_path),
                             key=lambda x: [int(part) if part.isdigit() else part for part in re.split('([0-9]+)', x)])
        for image_file in image_files:
            image_paths.append(os.path.join(folder_path, image_file).replace('\\', '/'))
dataset = fused_CustomDataset(image_paths, second_features, image_labels, transform=transform)

# split the dataset
train_size = int(0.5 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                         [train_size, val_size, test_size],
                                                                         generator=torch.Generator().manual_seed(42))

# dataloader
batch_size = 20
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# define the Fully Convolutional Network (FCN) model and optimizer
num_classes = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCN(batch_size, num_classes=num_classes, second_features=second_features)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion: CrossEntropyLoss = nn.CrossEntropyLoss()

num_epochs = 5

train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, second_features, labels in train_loader:
        images, labels = images, labels
        optimizer.zero_grad()
        outputs = model(images, second_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        train_loss += loss.item() * images.size(0)
        train_correct += torch.sum(preds == labels.data)


    train_loss = train_loss / len(train_dataset)
    train_acc = train_correct / len(train_dataset)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, second_features, labels in val_loader:
            images, labels = images, labels

            outputs = model(images, second_features)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * images.size(0)
            val_correct += torch.sum(preds == labels.data)

    val_loss = val_loss / len(val_dataset)
    val_acc = val_correct / len(val_dataset)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc * 100:.4f}%")
    print(f"Val Loss: {val_loss:.4f} Val Acc: {val_acc * 100:.4f}%")
    print("-" * 20)


torch.save(model.state_dict(), './fused_model.pth')
print("Model saved successfully.")
import matplotlib.pyplot as plt

# Assuming that the training and validation loss values are recorded in the lists train_loss_list and val_loss_list
# Make sure that `train_loss_list` and `val_loss_list` have a length of 15.
assert len(train_loss_list) == len(val_loss_list) == num_epochs, "Length of loss list doesn't match num_epochs"

# Plot the loss curves for the training set and validation set.
plt.figure()
plt.plot(range(1, num_epochs + 1), train_loss_list, label='train')
plt.plot(range(1, num_epochs + 1), val_loss_list, label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.savefig('./fused_loss_plot.png')

plt.figure()
plt.plot(range(1, num_epochs + 1), train_acc_list, label='train')
plt.plot(range(1, num_epochs + 1), val_acc_list, label='validation')
plt.xlabel('Epoch')
plt.ylabel('Model Accuracy')
plt.title('fused_Accuracy in Train dataset and Validation dataset')
plt.legend()
plt.savefig('./fused_accuracy_plot.png')

#plt.show()
