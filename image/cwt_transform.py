import os
import matplotlib.pyplot as plt
import matplotlib
import scaleogram as scg
import pandas as pd
import numpy as  np
#  specify the path of the source folder and the target folder
source_folder_path = './dataset/day2/'
output_folder_path = './test/2'


# Get the file list from the source folder
file_list = os.listdir(source_folder_path)

# loop through each file in the file list obtained from the source folder
for file in file_list:
    # Construct the complete path of the source file.
    source_file_path = os.path.join(source_folder_path, file)
    data = pd.read_csv(source_file_path)
    # Extract columns x and y.
    x_values = data.iloc[:, 0]
    y_values = data.iloc[:, 1]
    # Set the scale.
    scales = scg.periods2scales(np.arange(1, 40))
    # Plot the data.
    ax2 = scg.cws(time=x_values.to_numpy(), signal=y_values, scales=scales, figsize=(7, 2))

    plt.xlabel('wave number (cm-1)')  # 修改横坐标标签为"波长"
    plt.ylabel('absorbancy')  # 修改纵坐标标签为"强度"

    plt.tight_layout()

    # construct the complete path for the target file
    output_file_path = os.path.join(output_folder_path, file.split('.')[0] + '.jpg')
    plt.savefig(output_file_path, dpi=100)
    plt.close()
