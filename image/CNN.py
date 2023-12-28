import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
# pre-processing
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


# define model
class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.feature_extractor = models.resnet50(pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # 检查输入维度
        if x.dim() == 3:
            x = x.unsqueeze(0)

        features = self.feature_extractor(x)
        features = features.unsqueeze(2).unsqueeze(3)
        pooled_features = self.avg_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifier(pooled_features)
        return output



data_path = '/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/fused_data/bloodstain/bloodstain/image/cwt/'


# Load and split the dataset
dataset = ImageFolder(data_path, transform=train_transform)
train_size = int(0.5 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset_CNN, val_dataset_CNN, test_dataset = torch.utils.data.random_split(dataset,
                                                                                 [train_size, val_size, test_size],
                                                                                 generator=torch.Generator().manual_seed(
                                                                                     42))

train_dataset_CNN.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform
val_dataset_CNN.dataset.transform = val_transform

# Define data loaders
train_loader = DataLoader(train_dataset_CNN, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)
val_loader = DataLoader(val_dataset_CNN, batch_size=20, shuffle=False)

# Define label names
label_names = dataset.classes
print(label_names)
# Get a batch of data and labels
data_iter = iter(test_loader)
images, labels = next(data_iter)


# Define FCN model and optimizer
'''device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCN(len(label_names)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 50
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
#training and validation
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        train_loss += loss.item() * images.size(0)
        train_correct += torch.sum(preds == labels.data)

    train_loss = train_loss / len(train_dataset_CNN)
    train_acc = 100*train_correct / len(train_dataset_CNN)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc.item())

    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * images.size(0)
            val_correct += torch.sum(preds == labels.data)

    val_loss = val_loss / len(val_dataset_CNN)
    val_acc = 100* val_correct / len(val_dataset_CNN)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc.item())

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}%")
    print(f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}%")
    print("-" * 20)

# save the model
torch.save(model.state_dict(), '/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/fused_data/bloodstain/图像分类/model.pth')
print("Model saved successfully.")
import matplotlib.pyplot as plt

# Assuming that the training and validation loss values are recorded in the lists train_loss_list and val_loss_list

# assert the length of train_loss_list and val_loss_list
assert len(train_loss_list) == len(val_loss_list) == num_epochs, "Length of loss list doesn't match num_epochs"

# plot curve
plt.plot(range(1, num_epochs + 1), train_loss_list, label='train')
plt.plot(range(1, num_epochs + 1), val_loss_list, label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()  
plt.savefig('/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/图片分类结果/loss_plot.png')
plt.show()


plt.plot(range(1, num_epochs + 1), train_acc_list, label='train')
plt.plot(range(1, num_epochs + 1), val_acc_list, label='validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0.6, 1.0)
plt.title('Model Accuracy')
plt.legend()  # add legend
plt.savefig('/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/图片分类结果/accuracy_plot.png')
plt.show()
'''
