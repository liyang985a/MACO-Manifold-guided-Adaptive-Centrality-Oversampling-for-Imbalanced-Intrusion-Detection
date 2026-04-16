import numpy as np

class_names = np.load('label_encoder_classes.npy', allow_pickle=True)
for i, name in enumerate(class_names):
    print(f"Label {i}: {name}")
