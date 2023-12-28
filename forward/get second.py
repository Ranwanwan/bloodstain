import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

# define dataset
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


data_dir = '/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/fused_data/bloodstain/bloodstain/image classification/dataset/'  # 血迹文件夹的路径
label_map = {'day1': 0, 'day2': 1, 'day3': 2, 'day4': 3, 'day5': 4, 'day6': 5, 'day7': 6}
all_data = []
all_labels = []

features = []
labels = []

file_count = 0  # counter
folders = ['day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'day7']

for folder in folders:
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        filenames = sorted(os.listdir(folder_path),
                             key=lambda x: [int(part) if part.isdigit() else part for part in re.split('([0-9]+)', x)])
        for filename in filenames:
            if filename.endswith('.txt'):
                file_count += 1
                file_path = os.path.join(folder_path, filename)
                data = pd.read_csv(file_path)
                wavelength = data.iloc[:, 0]
                intensity = data.iloc[:, 1]
                # normalization
                intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())

                second_derivative = np.gradient(np.gradient(intensity_normalized))
                # Create a tensor filled with zeros of shape [7] as the class feature.
                class_feature = torch.zeros(7)

                # Set the index position corresponding to the class feature to 1.
                class_feature[label_map[folder]] = 1
                combined_feature = torch.from_numpy(second_derivative).float()
                features.append(combined_feature)

                # features.append((second_derivative))
                labels.append(label_map[folder])

                print(f"Processing file: {file_path}")  # 打印当前处理的文件

tensor_data = [torch.from_numpy(np.column_stack(data)).float() for data in features]
tensor_labels = torch.tensor(labels)

dataset = MyDataset(tensor_data, tensor_labels)

output_dir = '/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/fused_data/bloodstain/bloodstain/forward feature fusion/'
os.makedirs(output_dir, exist_ok=True)
# data = torch.tensor(dataset.data)
label = dataset.labels
print("Second derivatives and labels saved successfully.")

import os
import torch

input_dir = '/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/fused_data/bloodstain/bloodstain/forward feature fusion/'
output_file = '/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/fused_data/bloodstain/bloodstain/forward feature fusion/file.pt'

combined_data = torch.stack(dataset.data)
combined_labels = torch.tensor(label)

combined_data_file = os.path.join(output_dir, 'normed_combined.pt')

data = {
    "data": combined_data,
    "label": combined_labels
}

torch.save(data, combined_data_file)
