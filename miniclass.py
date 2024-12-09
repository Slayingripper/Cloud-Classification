import requests
import cv2
import numpy as np
import tensorflow as tf
import numpy as np
import paho.mqtt.client as mqtt
import configparser
import math

# Read configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Extract MQTT settings
mqtt_server = config['MQTT']['server']
mqtt_port = int(config['MQTT']['port'])
mqtt_topic = config['MQTT']['topic']

# Extract model path
model_path = config['Model']['path']

# Extract image URL
image_url = config['Image']['url']

# Step 1: Fetch the image
response = requests.get(image_url)
client = mqtt.Client()

# Connect to the MQTT broker
client.connect(mqtt_server, mqtt_port, 60)

if response.status_code == 200:
    img_array = np.asarray(bytearray(response.content), dtype="uint8")
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
else:
    print(f"Failed to fetch image: {response.status_code}")
    exit()

# Step 2: Preprocess the image for classification
img_resized = cv2.resize(img, (224, 224))  # Resize to fit the model input
img_normalized = img_resized / 255.0       # Normalize pixel values to [0, 1]

# Step 3: Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path)
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
    'Altostratus',    # As
    'Cumulus',        # Cu
    'Cumulonimbus',   # Cb
    'Nimbostratus',   # Ns
    'Stratocumulus',  # Sc
    'Stratus',        # St
    'Contrail'        # Ct
]  # 11 classes

predicted_cloud = None
if output_data.size > 0:
    # Get the predicted cloud class
    predicted_class = np.argmax(output_data, axis=1)

    if predicted_class.size > 0:
        predicted_cloud = cloud_classes[predicted_class[0]]
        print(f"Predicted Cloud Class: {predicted_cloud}")
           
        # Publish the predicted cloud class to the MQTT topic
        client.publish(mqtt_topic, predicted_cloud)
    else:
        print("No predicted class available.")
else:
    print("No output data available.")

# --------------------------------------------
# Estimate wind direction from cloud patterns
# --------------------------------------------

# We will use a heuristic approach:
# 1. Detect edges (Canny).
# 2. Find lines using Hough transform.
# 3. Determine average angle of these lines.
# 4. Infer wind direction based on the line orientation.

if predicted_cloud is not None:
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply a blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is not None:
        # Compute average angle of all detected lines
        angles = []
        for line in lines:
            for rho, theta in line:
                # Convert angle from radians to degrees
                angle_deg = (theta * 180 / np.pi)
                angles.append(angle_deg)
        
        if len(angles) > 0:
            avg_angle = np.mean(angles)
            print(f"Average angle of cloud lines: {avg_angle:.2f} degrees")

            # Now we interpret this angle:
            # Angle reference: 0° lines run vertically (y-axis), 90° lines run horizontally (x-axis)
            # We'll normalize angle around 0-180 range.
            # We can define a rough mapping from angle to direction:
            # For simplicity:
            #  - 0° ~ North-South lines
            #  - 90° ~ East-West lines
            # Wind direction often is perpendicular to the cloud alignment, but this is a big assumption.
            # We'll assume that elongated clouds form along the direction of the wind or slightly shifted.
            # We'll just map angles to compass directions.

            # Normalize the angle to [0,180) for ease of interpretation
            normalized_angle = avg_angle % 180

            # Define a simple heuristic:
            # If the lines are ~0 or ~180 degrees, they are aligned N-S.
            # If they are ~90 degrees, aligned E-W.
            # Values in between indicate diagonal directions.
            # We'll pick the direction where the wind "moves along" the lines. 
            # In reality, it's often perpendicular or influenced by larger scale patterns.
            # Here, we choose a simplistic interpretation:
            
            # We'll define direction sectors:
            #   0° ± 22.5° => North-South (N->S)
            #  22.5°-67.5° => NE-SW (NE->SW)
            #  67.5°-112.5° => East-West (E->W)
            # 112.5°-157.5° => SE-NW (SE->NW)
            # >157.5° back towards N-S

            if normalized_angle < 22.5 or normalized_angle >= 157.5:
                wind_direction = "North to South"
            elif 22.5 <= normalized_angle < 67.5:
                wind_direction = "NorthEast to SouthWest"
            elif 67.5 <= normalized_angle < 112.5:
                wind_direction = "East to West"
            elif 112.5 <= normalized_angle < 157.5:
                wind_direction = "SouthEast to NorthWest"
            else:
                # Fallback, should not reach here due to conditions above
                wind_direction = "Undetermined"
            
            print(f"Inferred Wind Direction (Heuristic): {wind_direction}")
            # Publish wind direction as well
            client.publish(mqtt_topic, f"Wind Direction: {wind_direction}")
            #publish angle as well
            client.publish(mqtt_topic, f"Angle: {normalized_angle}")
        else:
            print("No lines detected to estimate wind direction.")
    else:
        print("No discernible linear cloud patterns detected. Wind direction not estimated.")
else:
    print("Cloud type not determined. Skipping wind direction estimation.")
