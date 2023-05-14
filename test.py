import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

x = tf.keras.Input(shape=(10,))

# Load the CSV file
df = pd.read_csv("file_fixed.csv")

# Sort by filename and class
df = df.sort_values(by=['filename', 'class'])

# Group by filename
grouped = df.groupby('filename')


X = df['filename']
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load image data from directory using TensorFlow
data_dir = pathlib.Path('Testing')
batch_size = 100
img_height = 512
img_width = 512

print()
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Print class names
class_names = train_ds.class_names
print(class_names)

model = Sequential([
    Conv2D(16,(3,3), activation = 'relu', input_shape=(img_height,img_width,3)),
    MaxPooling2D((2,2)),
    Conv2D(16,(3,3), activation = 'relu'),
    Flatten(),
    Dense(32,activation = 'relu'),
    Dense(1,activation = 'sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,epochs=1, validation_data=val_ds)

idx2 = random.randint(0,len(y_test))
plt.imshow(X_test[idx2,:])
plt.show()
 
y_pred = model.predict(X_test[idx2,:].reshape(256))

y_pred = y_pred > .5

if(y_pred==0):
    print("Dog")
else:
    print("Broken")