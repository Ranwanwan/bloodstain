import torch
import pandas as pd
from multi.数据增强.fused_data.bloodstain.bloodstain.image.CNN import FCN, test_dataset

num_classes = 7
model = FCN(num_classes=num_classes)
model.load_state_dict(torch.load('/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/fused_data/bloodstain/GUI/model.pth', map_location=torch.device('cpu')))
#model from image classification-CNN
model.eval()
output_list = []
labels_list = []

for image, label in test_dataset:
    image = image.unsqueeze(0)

    # Use the model for inference.
    with torch.no_grad():
        output = model(image)
        # get predicted result
        probabilities = torch.softmax(output, dim=1)
        _, predicted_class = torch.max(probabilities, 1)
        predicted_label = predicted_class.item()
        output_list.append(output)
        labels_list.append(predicted_label)

# Combine output results and labels into one DataFrame
df = pd.DataFrame({'Output': output_list, 'Label': labels_list})

# save as CSV file
df.to_csv('/Users/liran/Desktop/IMage-Seq-Text/multi/数据增强/fused_data/bloodstain/bloodstain/backward/test_output.CSV', index=False)
