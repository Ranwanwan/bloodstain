import numpy as np
import matplotlib.pyplot as plt
import torch

from CNN import FCN,test_dataset


# Create a zero matrix of size num_classes x num_classes
num_classes = 7
confusion_matrix = np.zeros((num_classes, num_classes))

# Load the model
model = FCN(num_classes=num_classes)
model.load_state_dict(torch.load('./model.pth', map_location=torch.device('cpu')))
model.eval()

# Iterate through the test dataset samples
for image, label in test_dataset:
    # Convert the image to a format accepted by the model
    image = image.unsqueeze(0)

    # Perform inference using the model
    with torch.no_grad():
        output = model(image)

    # Get the predicted result
    probabilities = torch.softmax(output, dim=1)
    _, predicted_class = torch.max(probabilities, 1)
    predicted_label = predicted_class.item()

    # Increase the count at the corresponding position in the confusion matrix
    confusion_matrix[label][predicted_label] += 1

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix)
# Visualize the confusion matrix
plt.imshow(confusion_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()

class_names = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7']
# Print class labels

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

plt.xlabel('Predicted Class')
plt.ylabel('True Class')

for i in range(len(confusion_matrix)):
    for j in range(len(confusion_matrix)):
        plt.text(j, i, int(confusion_matrix[i, j]), ha='center', va='center', color='black')

# Save the confusion matrix image
#plt.savefig('/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/fused_data/bloodstain/图像分类/newtest_image_confusion_Matrix.png', dpi=350)
plt.show()

# Calculate precision
precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
print('Precision:', precision)

# Calculate recall
recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
print('Recall:', recall)

# Calculate F1score
f1 = 2 * (precision * recall) / (precision + recall)
print('F1 Score:', f1)
# Calculate accuracy
accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
print('Accuracy:', accuracy)

