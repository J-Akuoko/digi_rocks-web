import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import PIL
import numpy as np
import cv2


TF_ENABLE_ONEDNN_OPTS=0

import os
from PIL import Image

im = Image.open('test/immmm1.jpg')

# Resize the image to (150, 150)
img = im.resize((150, 150))

# Convert the image to a NumPy array
img_array = np.array(img)

# Convert the image from RGB to a NumPy array (if it's already RGB, this step is not needed)
# img_array = img_array[:, :, ::-1]  # Uncomment if the image is in BGR format

# Normalize the image
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Check the shape of the processed image
#print(f'Processed Image Shape: {img_array.shape}')

import os
from PIL import Image

# Path to the data directory
data_dir = 'data'

# Convert .jfif to .jpg
for subdir in os.listdir(data_dir):
    subpath = os.path.join(data_dir, subdir)
    if os.path.isdir(subpath):
        for file in os.listdir(subpath):
            if file.endswith('.jfif'):
                file_path = os.path.join(subpath, file)
                image = Image.open(file_path)
                new_file_path = os.path.splitext(file_path)[0] + '.jpg'
                image.save(new_file_path, 'JPEG')
                os.remove(file_path)  # Remove the original .jfif file

print("Conversion complete")




train_data = image_dataset_from_directory(
    'data',
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(150, 150),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=True
    )

model = Sequential([
    Input(shape = (150,150,3)),
    Conv2D(64, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = 'relu'),
    MaxPooling2D(),
    Conv2D(32, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = 'relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(16, activation = 'relu'),
    Dense(8, activation = 'relu'),
    Dense(2, activation = 'softmax')
])

model.compile(loss = 'SparseCategoricalCrossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(train_data, epochs = 100)

instance = model.predict(img_array)
new_instance = instance.reshape(-1,)

if new_instance[0].item() > 0.5:
    print('The prediction is a cat')
else:
    print('The prediction is a dog')