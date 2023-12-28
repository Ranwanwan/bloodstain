from torch.utils.data import Dataset, DataLoader

import os
from PIL import Image
from torch.utils.data import Dataset


class fused_CustomDataset(Dataset):
    def __init__(self, image_paths, second_features, labels, transform):
        self.image_paths = image_paths
        self.second_features = second_features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        second_feature = self.second_features[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, second_feature, label[0]
        # else:
        #     print(f"The path '{image_path}' is not a valid folder.")
        #     return None
