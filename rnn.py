import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Flatten, TimeDistributed, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define image dimensions and other parameters
IMG_HEIGHT, IMG_WIDTH = 256, 256
NUM_CLASSES = 21
BATCH_SIZE = 32
EPOCHS = 100

# Path to dataset
dataset_dir = 'C:\Users\HP\OneDrive - vit.ac.in\Desktop\VIT\Deep Learning Project\UCMerced_LandUse\Images'

# Function to load images and labels
def load_images_and_labels(dataset_dir, img_height, img_width):
    images = []
    labels = []
    classes = sorted(os.listdir(dataset_dir))
    for label, class_name in enumerate(classes):
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.endswith('.tif'):
                    img_path = os.path.join(class_path, img_name)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (img_width, img_height))
                    images.append(img)
                    labels.append(label)  # Add label
    # Adjust labels to range 0-20 if they are 1-21
    labels = [l - 1 for l in labels]
    return np.array(images), np.array(labels), classes

# Load images and labels
images, labels, class_names = load_images_and_labels(dataset_dir, IMG_HEIGHT, IMG_WIDTH)

# Normalize images
images = images / 255.0

# Reshape images for LSTM input (Treat each row as a time step)
images = images.reshape(-1, IMG_HEIGHT, IMG_WIDTH * 3)  # Each row of 256x3 pixels is a time step

# Convert labels to categorical
labels = to_categorical(labels, NUM_CLASSES)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Define LSTM model architecture
model = Sequential([
    LSTM(128, activation='tanh', input_shape=(IMG_HEIGHT, IMG_WIDTH * 3), return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(128, activation='tanh', return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')  # Softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model architecture
model.summary()

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val)
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_accuracy:.4f}')
