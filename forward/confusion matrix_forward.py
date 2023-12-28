import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import confusion_matrix
from sklearn import model_selection as ms
from FCN import FCN
from dataloader import fused_CustomDataset


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

l = torch.load('./normed_combined.pt')
second_features = l['data']
second_labels = l['label']

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


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
])

data_path = './cwt/'
second_features = second_features

# Create an empty list to store image labels
image_labels = []

# Iterate through the files in the data path
for file in os.listdir(data_path):
    folder_path = os.path.join(data_path, file)

    # Check if the file is a directory
    if os.path.isdir(folder_path):
        # Extract the label from the directory name
        label = int(file) - 1

        # Get the list of files in the directory
        label_files = os.listdir(folder_path)

        # Append the label and image file name to the image_labels list
        image_labels += [(label, image_file) for image_file in label_files]

# Create an empty list to store the image paths
image_paths = []

# Iterate through the files in the data path again
for file in os.listdir(data_path):
    folder_path = os.path.join(data_path, file)

    # Check if the file is a directory
    if os.path.isdir(folder_path):
        # Get the list of files in the directory
        image_files = os.listdir(folder_path)

        # Iterate through the image files
        for image_file in image_files:
            # Append the full image path to the image_paths list, replacing backslashes with forward slashes
            image_paths.append(os.path.join(folder_path, image_file).replace('\\', '/'))

# Create a fused_CustomDataset object, passing in the image_paths, second_features, image_labels, and transform
dataset = fused_CustomDataset(image_paths, second_features, image_labels, transform=transform)


train_size = int(0.5 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                         [train_size, val_size, test_size],
                                                                         generator=torch.Generator().manual_seed(42))

# dataloader
batch_size = 20
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

num_classes = 7
# confusion_matrix = np.zeros((num_classes, num_classes))


model = FCN(batch_size, num_classes=num_classes, second_features=second_features)
model.load_state_dict(torch.load('./model/fused_model.pth'))
model.eval()

true_labels = []
predicted_labels = []
conf_matrix = np.zeros((num_classes, num_classes))

for image, second_feature, label in test_loader:


    with torch.no_grad():
        output = model(image, second_feature)

    _, predicted = torch.max(output.data, 1)
    conf_matrix += confusion_matrix(label, predicted, labels=np.arange(num_classes))


class_names = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7']

print('conf_matrix:')
print(conf_matrix)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')

plt.colorbar()
plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('forward_confusion_matrix',dpi=350)
plt.show()
# calculate precision
precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
precision_avg = np.mean(precision)
print('Precision:', precision)
print('Average Precision:', precision_avg)

# recall
recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
recall_avg = np.mean(recall)
print('Recall:', recall)
print('Average Recall:', recall_avg)

# F1score
f1 = 2 * (precision * recall) / (precision + recall)
f1_avg = np.mean(f1)
print('F1 Score:', f1)
print('Average F1 Score:', f1_avg)
# accuracy
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
print('Accuracy:', accuracy)
