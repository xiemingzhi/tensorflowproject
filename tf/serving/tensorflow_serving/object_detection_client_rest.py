from __future__ import print_function

import base64
import requests
import json 
import numpy as np

# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://192.168.99.100:8501/v1/models/object_detection:predict'

# The image URL is the location of the image we should send to the server
#IMAGE_URL = 'cat.jpg'
IMAGE_URL = 'clock.jpg'

def main():
  # Download the image
  #dl_request = requests.get(IMAGE_URL, stream=True)
  #dl_request.raise_for_status()

  # Compose a JSON Predict request (send JPEG image in base64).
  #jpeg_bytes = base64.b64encode(dl_request.content).decode('utf-8')
  with open(IMAGE_URL, "rb") as img_file:
    jpeg_bytes = base64.b64encode(img_file.read()).decode('utf-8')
  instance = [{"b64": jpeg_bytes}]

  #predict_request = '{"signature_name": "serving_default", "instances" : [{"b64": "%s"}]}' % jpeg_bytes
  predict_request = json.dumps({"instances": instance})

  headers = {"content-type": "application/json"}

  # Send few requests to warm-up the model.
  for _ in range(3):
    response = requests.post(SERVER_URL, data=predict_request, headers=headers)
    response.raise_for_status()

  # Send few actual requests and report average latency.
  total_time = 0
  num_requests = 10
  for _ in range(num_requests):
    response = requests.post(SERVER_URL, data=predict_request, headers=headers)
    response.raise_for_status()
    total_time += response.elapsed.total_seconds()
    prediction = response.json()['predictions'][0]

  #print('Prediction class: {}, avg latency: {} ms'.format(
  #    prediction['classes'], (total_time*1000)/num_requests))
  print('Prediction class: {}, avg latency: {} ms'.format(
      prediction['detection_classes'], (total_time*1000)/num_requests))

if __name__ == '__main__':
  main()