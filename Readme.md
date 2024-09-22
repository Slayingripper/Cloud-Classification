# Cloud Classification Project

This project classifies cloud types from images using a TensorFlow Lite model and publishes the results to an MQTT broker. This project is used in conjuction with the CCSN dataset and an ALLSKY camera.

## Table of Contents
- [Cloud Classification Project](#cloud-classification-project)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Run this periodically](#run-this-periodically)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
    ```
2. Install the required Python packages:
    ```
   pip install -r requirements.txt
    ```

## Configuration

Edit the config.ini file to set up the project:

    MQTT Settings: Configure the MQTT broker.
        server: IP address of the MQTT server.
        port: Port number of the MQTT server.
        topic: MQTT topic to publish the predictions.

    Model Settings: Path to the TensorFlow Lite model file.
        path: Path to the .tflite model file.

    Image Settings: URL of the image to classify.
        url: URL of the image.    

## Usage

Run the miniclass.py script to classify the cloud type from the image and publish the result to the MQTT broker:
    
    
        python miniclass.py
        
### Run this periodically 

To run the miniclass.py script periodically using cron, follow these steps:

Open the crontab editor:

    crontab -e

Add a new cron job to run the script at your desired interval. For example, to run the script every hour, add the following line:


    0 * * * * /usr/bin/python3 /path/to/your/project/miniclass.py

Make sure to replace /usr/bin/python3 with the path to your Python interpreter and /path/to/your/project/miniclass.py with the actual path to your script.

Save and close the crontab editor.