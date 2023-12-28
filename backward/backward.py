import csv
import pprint
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

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

imputer = SimpleImputer(strategy='mean')  # 或者使用其他的填充策略，如'median', 'most_frequent'等
features = imputer.fit_transform(features)

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# split dataset
imputer = SimpleImputer(strategy='mean')
features = imputer.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=42)
print(len(X_test))

# Create objects for Random Forest, Support Vector Machine, and K-Nearest Neighbors classifiers.
rf_clf = RandomForestClassifier()
svm_clf = SVC()
knn_clf = KNeighborsClassifier()
dt_clf = DecisionTreeClassifier()

# RF
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, average=None)
rf_recall = recall_score(y_test, rf_pred, average=None)
rf_f1score = f1_score(y_test, rf_pred, average=None)
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"Random Forest Precision: {rf_precision}")
print(f"Random Forest Recall: {rf_recall}")
print(f"Random Forest F1-score: {rf_f1score}")

# KNN
knn_clf.fit(X_train, y_train)
knn_pred = knn_clf.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred, average=None)
knn_recall = recall_score(y_test, knn_pred, average=None)
knn_f1score = f1_score(y_test, knn_pred, average=None)
print(f"KNN Accuracy: {knn_acc:.4f}")
print(f"KNN Precision: {knn_precision}")
print(f"KNN Recall: {knn_recall}")
print(f"KNN F1-score: {knn_f1score}")

# DT
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred, average=None)
dt_recall = recall_score(y_test, dt_pred, average=None)
dt_f1score = f1_score(y_test, dt_pred, average=None)
print(f"Decision Tree Accuracy: {dt_acc:.4f}")
print(f"Decision Tree Precision: {dt_precision}")
print(f"Decision Tree Recall: {dt_recall}")
print(f"Decision Tree F1-score: {dt_f1score}")



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Create a Random Forest classifier object.
rf = RandomForestClassifier()
dt =  DecisionTreeClassifier()
knn =  KNeighborsClassifier()
# Calculate the validation accuracy of the model using cross-validation.
rf_scores = cross_val_score(rf, features, labels, cv=7)  # 7折交叉验证
dt_scores = cross_val_score(dt, features, labels, cv=7)
knn_scores = cross_val_score(knn, features, labels, cv=7)
print("Validation Accuracy: {:.4f}".format(rf_scores.mean()))
print("Validation Accuracy: {:.4f}".format(dt_scores.mean()))
print("Validation Accuracy: {:.4f}".format(knn_scores.mean()))
