import os
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Set the path to your dataset
base_dir = 'C:/Users/jeevanantham/Desktop/plantling_cnn/plantling_dataset'

# Image Data Generator for data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load data
data_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'  # Automatically assign classes based on subfolder names
)

# Calculate steps per epoch
steps_per_epoch = len(data_generator)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(data_generator.class_indices), activation='softmax')  # Number of classes from the dataset
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    data_generator,
    epochs=10,  # Adjust this number as needed
    steps_per_epoch=steps_per_epoch,
)

# Save the model
model.save('seedling_classifier.h5')

# Save class indices to a JSON file
class_indices = data_generator.class_indices
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)
