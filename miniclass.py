import requests
import cv2
import numpy as np
import tensorflow as tf

# Step 1: Fetch the image
url = 'http://172.25.63.4/current/tmp/image.jpg'
response = requests.get(url)

if response.status_code == 200:
    img_array = np.asarray(bytearray(response.content), dtype="uint8")
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
else:
    print(f"Failed to fetch image: {response.status_code}")
    exit()

# Step 2: Preprocess the image
img_resized = cv2.resize(img, (224, 224))  # Resize to fit the model input
img_normalized = img_resized / 255.0       # Normalize pixel values to [0, 1]

# Step 3: Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='ccsn_cloud_classification_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare the input data
input_data = np.expand_dims(img_normalized, axis=0).astype(np.float32)  # Add batch dimension

# Set the tensor to the input data
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output data
output_data = interpreter.get_tensor(output_details[0]['index'])

# Check the output data
print("Output data shape:", output_data.shape)
print("Raw output data:", output_data)

# Define cloud classes
cloud_classes = [
    'Cirrus',         # Ci
    'Cirrostratus',   # Cs
    'Cirrocumulus',   # Cc
    'Altocumulus',    # Ac
    'Altostratus',     # As
    'Cumulus',        # Cu
    'Cumulonimbus',   # Cb
    'Nimbostratus',   # Ns
    'Stratocumulus',  # Sc
    'Stratus',        # St
    'Contrail'       # Ct
]  # 11 classes

if output_data.size > 0:
    # Get the predicted cloud class
    predicted_class = np.argmax(output_data, axis=1)

    if predicted_class.size > 0:
        predicted_cloud = cloud_classes[predicted_class[0]]
        print(f"Predicted Cloud Class: {predicted_cloud}")
    else:
        print("No predicted class available.")
else:
    print("No output data available.")
