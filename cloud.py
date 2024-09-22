import requests
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import models, layers

# Step 1: Fetch the image
url = 'http://172.25.63.4/current/tmp/image.jpg'
response = requests.get(url)
img_array = np.asarray(bytearray(response.content), dtype="uint8")
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# Step 2: Preprocess the image
img_resized = cv2.resize(img, (224, 224))  # Resize to fit the model input
img_normalized = img_resized / 255.0       # Normalize pixel values to [0, 1]

# Step 3: Load the pre-trained MobileNetV2 model with custom layers for cloud classification
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers for classification
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')  # Assuming 5 cloud classes; adjust accordingly
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the trained weights (replace 'cloud_model_weights.h5' with your model's weights)
model.load_weights('ccsn_cloud_classification_model.h5')

# Step 4: Make predictions
predictions = model.predict(np.expand_dims(img_normalized, axis=0))

# Get the predicted cloud class
predicted_class = np.argmax(predictions, axis=1)

# You can also map the class index to class names
cloud_classes = ['Cirrus', 'Cumulus', 'Stratus', 'Altostratus', 'Nimbostratus']  # Example classes
predicted_cloud = cloud_classes[predicted_class[0]]

print(f"Predicted Cloud Class: {predicted_cloud}")

