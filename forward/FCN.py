# 定义FCN模型
import torch
from torch import nn
from torchvision import models


class FCN(nn.Module):
    def __init__(self, batch_size, num_classes, second_features):
        super(FCN, self).__init__()
        self.bs = batch_size
        resnet50 = models.resnet50(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        num_features = resnet50.fc.in_features
        # self.feature_extractor.fc = nn.Identity()
        self.second_features = second_features
        combined_features_dim = num_features + self.second_features.shape[2]  # 计算组合后的特征维度
        self.classifier = nn.Linear(combined_features_dim, num_classes)

    def forward(self, x, s_features):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        s_features = s_features.view(self.bs, -1)
        features = self.feature_extractor(x)
        features = features.unsqueeze(2).unsqueeze(3)
        pooled_features = self.avg_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        pooled_features = pooled_features.view(self.bs, 2048)  # 调整维度
        # Concatenate the second-order derivative data with the image feature.
        combined_features = torch.cat((pooled_features, s_features), dim=-1)
        output = self.classifier(combined_features)
        return output