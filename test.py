import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split



# Load the CSV file
df = pd.read_csv("file_fixed.csv")

# Sort by filename and class
df = df.sort_values(by=['filename', 'class'])

# Group by filename
grouped = df.groupby('filename')

# Iterate over groups and print filenames and classes


# Split the dataset into train and test sets

X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load image data from directory using TensorFlow
data_dir = pathlib.Path('Testing')
batch_size = 100
img_height = 512
img_width = 512
print("FILE")
print(data_dir)
print("FILE")
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

