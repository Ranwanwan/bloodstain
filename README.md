# bloodstain
### Install 
```bash
pip install -r requirements.txt
```
Image Classification: 
```bash
First, use cwt.py to convert the dataset data into images. 
Please refer to the tutorial video for specific instructions. 
Once you have obtained the corresponding images, you can use CNN.py for classification.
```
forward feature fusion:
```bash
First, run `get_second.py` to obtain `combined.pt`. Then, run `training.py` and `confusion_matrix` to get the results.
```
backward feature fusion:
```bash
Run `test_output.py` to obtain `test_output`. Once obtained, open Excel and perform the required operations (as explained in the video).
Finally, concatenate the obtained `second_features` and `second_labels` with `test_output`, classify them in the backward phase to obtain the results, and finally calculate the accuracy using `confusion_matrix`.
```
