import torch
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
l=torch.load('/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/fused_data/bloodstain/bloodstain/forward/normed_combined.pt')
second_features = l['data']
second_labels = l['label']
features_array = second_features.numpy()
labels_array = second_labels.numpy()

reshaped_features = np.reshape(features_array, (features_array.shape[0], -1))

#df_features = pd.DataFrame(second_features)
#df_labels = pd.DataFrame(labels_array)
np.set_printoptions(suppress=True, precision=6)
np.savetxt('/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/fused_data/bloodstain/bloodstain/backward/second_features.csv', reshaped_features, delimiter=',')
np.savetxt('/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/fused_data/bloodstain/bloodstain/backward/second_labels.csv',labels_array,delimiter=',')
