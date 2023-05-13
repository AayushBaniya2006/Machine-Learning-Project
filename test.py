import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# Load the CSV file
df = pd.read_csv('export/_annotations.csv')

# Preprocess the dataset
df = df[['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax']]
df['class'] = df['class'].astype('category').cat.codes
df['filename'] = df['filename'].apply(lambda x: 'path/to/images/' + x)
df['width'] = 512
df['height'] = 512

# Split the dataset
from sklearn.model_selection import train_test_split

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


data_dir = pathlib.Path("path/to/images")

batch_size = 100
img_height = 512
img_width = 512

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

