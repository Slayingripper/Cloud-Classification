import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint

# Define paths
train_dir = '/home/minifalcon/Weather/CCSN_v2'  # Update with the path to your CCSN train data
val_dir = '/home/minifalcon/Weather/CCSN_v2'  # Update with the path to your CCSN validation data
model_save_path = 'ccsn_cloud_model_weights.keras'


# Step 1: Image Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize to fit the MobileNetV2 input
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Step 2: Create Model using Transfer Learning (MobileNetV2 as base model)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')  # number of classes detected automatically
])

# Step 3: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Set up checkpoint to save the model weights
checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, mode='min')

# Step 5: Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=10,  # Adjust based on your needs
    callbacks=[checkpoint]
)

# Step 6: Save the model structure as well
model.save('ccsn_cloud_classification_model.h5')

print(f"Model training complete. Weights saved to {model_save_path}")
