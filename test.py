from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

dataset = datasets.load_files("path/to/car/images", shuffle=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)

# Extract features from the images using HOG
from skimage.feature import hog
from skimage import color
import numpy as np

def extract_features(X):
    features = []
    for img in X:
        gray_img = color.rgb2gray(img)
        features.append(hog(gray_img, block_norm='L2', pixels_per_cell=(32, 32), cells_per_block=(1, 1)))
    return np.array(features)

X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

# Train an SVM classifier on the extracted features
clf = SVC(kernel='linear')
clf.fit(X_train_features, y_train)

# Evaluate the model on the testing set
y_pred = clf.predict(X_test_features)
print(classification_report(y_test, y_pred))
