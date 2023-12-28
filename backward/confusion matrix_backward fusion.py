import csv
import pprint
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#load the data
output_data=pd.read_csv('/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/fused_data/bloodstain/bloodstain/backward/test_output.CSV', header=None)
data=output_data.iloc[:, 0:7]
scaler = MinMaxScaler()
output = scaler.fit_transform(data)
output_data.iloc[:, 0:7] = output
output_df=pd.DataFrame(output)
print(len(output_df))

out_labels = output_data.iloc[:,7]
#print(output)
print(len(out_labels))
second_data=pd.read_csv('/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/fused_data/bloodstain/bloodstain/backward/second_features.csv', header=None)
seco_output=second_data.iloc[:, 0:1763]
second_output_df=pd.DataFrame(second_data)
print(len(second_output_df))
#print(seco_labels)
features = pd.concat([output_df, second_output_df], axis=0)
print(len(features))
#print(labels)
labels_data=pd.read_csv('/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/fused_data/bloodstain/bloodstain/backward/second_labels.csv', header=None)
labels=pd.DataFrame(labels_data)
labels = pd.concat([out_labels, labels_data], axis=0)
print(len(labels))
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
features = imputer.fit_transform(features)

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#  Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=42)
print(len(X_test))
# Create objects for Random Forest, Support Vector Machine, and K-Nearest Neighbors classifiers.
rf_clf = RandomForestClassifier()
knn_clf = KNeighborsClassifier()
dt_clf = DecisionTreeClassifier()
# RF
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_acc:.4f}")


# KNN
knn_clf.fit(X_train, y_train)
knn_pred = knn_clf.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
print(f"KNN Accuracy: {knn_acc:.4f}")

# DT
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print(f"Decision Tree Accuracy: {dt_acc:.4f}")
from sklearn.metrics import confusion_matrix
rf_cm = confusion_matrix(y_test, rf_pred)
knn_cm = confusion_matrix(y_test, knn_pred)
dt_cm = confusion_matrix(y_test, dt_pred)
print("Random Forest Confusion Matrix:")
print(rf_cm)

print("KNN Confusion Matrix:")
print(knn_cm)

print("Decision Tree Confusion Matrix:")
print(dt_cm)
#class_names = ['0', '1', '2', '3', '4', '5', '6']
import matplotlib.pyplot as plt
import seaborn as sns
# 从 labels_data 获取类别标签
class_names = ['Day1','Day2','Day3','Day4','Day5','Day6','Day7']
# plot confusion matrix of RF
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Random Forest Confusion Matrix')
#plt.savefig('RF_CF.png',dpi=350)
plt.show()


# plot confusion matrix of KNN
plt.figure(figsize=(8, 6))
sns.heatmap(knn_cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('KNN Confusion Matrix')
#plt.savefig('KNN_CF.png',dpi=350)
#plt.show()

# # plot confusion matrix of DT
plt.figure(figsize=(8, 6))
sns.heatmap(dt_cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Decision Tree Confusion Matrix')
#plt.savefig('DT_CF.png',dpi=350)
#plt.show()
import  numpy as np
# calculate precision of RF within 7days
precision = np.diag(rf_cm) / np.sum(rf_cm, axis=0)
precision_avg = np.mean(precision)
print('Precision:', precision)
print('Average Precision:', precision_avg)

# calculate recall of RF within 7days
recall = np.diag(rf_cm) / np.sum(rf_cm, axis=1)
recall_avg = np.mean(recall)
print('Recall:', recall)
print('Average Recall:', recall_avg)

#  calculate recall of F1score within 7days
f1 = 2 * (precision * recall) / (precision + recall)
f1_avg = np.mean(f1)
print('F1 Score:', f1)
print('Average F1 Score:', f1_avg)
#  calculate accuracy of RF within 7days
accuracy = np.trace(rf_cm) / np.sum(rf_cm)
print('Accuracy:', accuracy)

