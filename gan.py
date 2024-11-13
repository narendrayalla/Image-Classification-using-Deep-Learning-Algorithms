import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D, UpSampling2D, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

# Parameters
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 256, 256, 3  # Adjust based on UC Merced Land Use images
LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 10000  # Set epochs as desired
DATASET_PATH = 'C:\Users\HP\OneDrive - vit.ac.in\Desktop\VIT\Deep Learning Project\UCMerced_LandUse\Images'

# Load images using ImageDataGenerator without train-test folder structure
datagen = ImageDataGenerator(rescale=1./255)
data_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode=None,  # No labels for GAN training
    shuffle=True
)

# Generator model
def build_generator():
    model = Sequential()
    model.add(Dense(128 * 64 * 64, activation="relu", input_dim=LATENT_DIM))
    model.add(Reshape((64, 64, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(IMG_CHANNELS, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    
    noise = Input(shape=(LATENT_DIM,))
    img = model(noise)
    return Model(noise, img)

# Discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    validity = model(img)
    return Model(img, validity)

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Build the generator
generator = build_generator()

# GAN model (combined generator and discriminator)
z = Input(shape=(LATENT_DIM,))
img = generator(z)
discriminator.trainable = False
validity = discriminator(img)
gan = Model(z, validity)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Training the GAN
for epoch in range(EPOCHS):
    # Train Discriminator
    imgs = next(data_generator)
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    gen_imgs = generator.predict(noise)
    
    real = np.ones((BATCH_SIZE, 1))
    fake = np.zeros((BATCH_SIZE, 1))
    
    d_loss_real = discriminator.train_on_batch(imgs, real)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train Generator
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    g_loss = gan.train_on_batch(noise, real)
    
    # Handle potential list output in g_loss
    g_loss_value = g_loss[0] if isinstance(g_loss, list) else g_loss
    
    # Print progress
    print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}] [G loss: {g_loss_value:.4f}]")
    
    # Optionally save generated images or model checkpoints
    if epoch % 1000 == 0:
        # Add saving logic here if desired
        pass
