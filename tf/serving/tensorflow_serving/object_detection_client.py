import grpc as implementations
import numpy as np
import scipy
import scipy.misc
import imageio
import matplotlib
import matplotlib.pyplot
import tensorflow as tf
import object_detection.utils.visualization_utils as vis_util
import object_detection.utils.label_map_util as label_map_util
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from keras.preprocessing import image
from PIL import Image
import inspect

tf.app.flags.DEFINE_string('server', '192.168.99.100:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_string('input_image', 'clock.jpg', 'Input Image clock.jpeg')
tf.app.flags.DEFINE_string('path_to_labels', 'D:/ming/git/tensorflow/models/research/object_detection/data/mscoco_complete_label_map.pbtxt', 'path to labels pbtxt')

FLAGS = tf.app.flags.FLAGS

# Create stub
host, port = FLAGS.server.split(':')
channel = implementations.insecure_channel(FLAGS.server)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Create prediction request object
request = predict_pb2.PredictRequest()

# Specify model name (must be the same as when the TensorFlow serving serving was started)
request.model_spec.name = 'object_detection'

# Initalize prediction 
# Specify signature name (should be the same as specified when exporting model)
#request.model_spec.signature_name = "detection_signature"
request.model_spec.signature_name = "serving_default"
# TODO determine size of image 
#img = image.load_img(FLAGS.input_image, target_size=(480,640))
#img = image.load_img(FLAGS.input_image, target_size=(425,425))
img = image.load_img(FLAGS.input_image, target_size=(640,640))
image2 = image.img_to_array(img)
image2 = image2.astype(np.uint8)
print("Input shape=",image2.shape )
request.inputs['inputs'].CopyFrom(
         tf.contrib.util.make_tensor_proto(image2, shape=[1] + list(image2.shape)))
	
# Call the prediction server
result = stub.Predict(request, 10.0)  # 10 secs timeout

# Plot boxes on the input image
# use util to load label map
category_index = label_map_util.create_category_index_from_labelmap(FLAGS.path_to_labels)
boxes = result.outputs['detection_boxes'].float_val
print("len boxes=", len(boxes))
classes = result.outputs['detection_classes'].float_val
print("len classes=", len(classes))
scores = result.outputs['detection_scores'].float_val
print("len scores=", len(scores))
#print(scores)

image_vis = vis_util.visualize_boxes_and_labels_on_image_array(
    image2,
    np.reshape(boxes,[100,4]),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8)

# Save inference to disk
imageio.imwrite('result.png', image_vis)
