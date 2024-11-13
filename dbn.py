import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# Parameters
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 256, 256, 3
NUM_CLASSES = 21
BATCH_SIZE = 32
DATASET_PATH = 'C:\Users\HP\OneDrive - vit.ac.in\Desktop\VIT\Deep Learning Project\UCMerced_LandUse\Images'

# Load dataset
datagen = ImageDataGenerator(rescale=1./255)
data_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Extract images and labels from generator and store in arrays
x_data, y_data = [], []
for _ in range(len(data_generator)):
    images, labels = next(data_generator)
    x_data.extend(images)
    y_data.extend(labels)

x_data = np.array(x_data)
y_data = np.array(y_data)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# Reshape images into 1D vectors for DBN processing
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# Scale data between 0 and 1
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(np.argmax(y_train, axis=1), NUM_CLASSES)
y_test = to_categorical(np.argmax(y_test, axis=1), NUM_CLASSES)

# Define custom DBN with stacked RBMs and a SoftMax classifier
class DBN(BaseEstimator, TransformerMixin):
    def __init__(self, rbm_layers=[256, 128], learning_rate=0.01, n_iter=10):
        self.rbm_layers = rbm_layers
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.rbms = [
            BernoulliRBM(n_components=n, learning_rate=self.learning_rate, n_iter=self.n_iter)
            for n in self.rbm_layers
        ]
        self.classifier = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')

    def fit(self, X, y):
        input_data = X
        for rbm in self.rbms:
            rbm.fit(input_data)
            input_data = rbm.transform(input_data)  # Pass transformed data to the next layer

        # Fine-tune with logistic regression
        self.classifier.fit(input_data, np.argmax(y, axis=1))
        return self

    def transform(self, X):
        for rbm in self.rbms:
            X = rbm.transform(X)
        return X

    def predict_proba(self, X):
        X_transformed = self.transform(X)
        return self.classifier.predict_proba(X_transformed)

    def predict(self, X):
        X_transformed = self.transform(X)
        return self.classifier.predict(X_transformed)

# Instantiate and train the DBN
dbn = DBN(rbm_layers=[512, 256], learning_rate=0.01, n_iter=20)
dbn.fit(x_train, y_train)

# Evaluate the DBN
y_pred = dbn.predict(x_test)
accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
print(f"DBN Accuracy: {accuracy:.4f}")
