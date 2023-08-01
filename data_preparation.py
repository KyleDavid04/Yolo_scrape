```python
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Define the path to the dataset and annotations
dataset_path = 'path_to_your_dataset'
annotations_path = 'path_to_your_annotations'

# Load the annotations
annotations = np.load(annotations_path)

# Get the list of image file names
image_files = os.listdir(dataset_path)

# Split the data into training and validation sets
train_files, val_files, train_annotations, val_annotations = train_test_split(image_files, annotations, test_size=0.2, random_state=42)

# Save the training and validation sets
np.save('train_files.npy', train_files)
np.save('val_files.npy', val_files)
np.save('train_annotations.npy', train_annotations)
np.save('val_annotations.npy', val_annotations)
```