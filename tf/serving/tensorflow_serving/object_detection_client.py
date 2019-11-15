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

#(tensorflow1) D:\ming\git\tensorflow\serving\tensorflow_serving>python object_detection_client.py --server=192.168.99.100:8500 --input_image=clock.jpeg --path_to_labels=D:/ming/git/tensorflow/models/research/object_detection/data/mscoco_complete_label_map.pbtxt
tf.app.flags.DEFINE_string('server', '192.168.99.100:8500', 'PredictionService host:port')
#tf.app.flags.DEFINE_string('input_image', 'clock.jpeg', 'Input Image clock.jpeg')
#tf.app.flags.DEFINE_string('input_image', 'coco-clock.png', 'Input Image clock.jpeg')
#tf.app.flags.DEFINE_string('input_image', 'coco-clock-1.png', 'Input Image clock.jpeg')
tf.app.flags.DEFINE_string('input_image', 'clock.jpg', 'Input Image clock.jpeg')
tf.app.flags.DEFINE_string('path_to_labels', 'D:/ming/git/tensorflow/models/research/object_detection/data/mscoco_complete_label_map.pbtxt', 'path to labels pbtxt')

FLAGS = tf.app.flags.FLAGS

# Create stub
host, port = FLAGS.server.split(':')
#channel = implementations.insecure_channel(host, int(port))
channel = implementations.insecure_channel(FLAGS.server)
#stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Create prediction request object
request = predict_pb2.PredictRequest()

# Specify model name (must be the same as when the TensorFlow serving serving was started)
request.model_spec.name = 'object_detection'

# Initalize prediction 
# Specify signature name (should be the same as specified when exporting model)
#request.model_spec.signature_name = "detection_signature"
request.model_spec.signature_name = "serving_default"
#request.inputs['inputs'].CopyFrom(
#          tf.contrib.util.make_tensor_proto(FLAGS.input_image))
#          tf.contrib.util.make_tensor_proto(matplotlib.pyplot.imread(FLAGS.input_image), shape=[1, 12000000]))
#          tf.contrib.util.make_tensor_proto(matplotlib.pyplot.imread(FLAGS.input_image), shape=[1, 427, 640, 1]))		  
#          tf.contrib.util.make_tensor_proto(scipy.misc.imread(FLAGS.input_image), shape=[1] + list(img.shape)))
#        tf.contrib.util.make_tensor_proto({FLAGS.input_image}))
#img = image.load_img(FLAGS.input_image, target_size=(480,640))
#img = image.load_img(FLAGS.input_image, target_size=(425,425))
img = image.load_img(FLAGS.input_image, target_size=(640,640))
image2 = image.img_to_array(img)
#image2 = image2.astype(np.float32)
image2 = image2.astype(np.uint8)
print("Input shape=",image2.shape )
#request.inputs['inputs'].CopyFrom(
#        tf.contrib.util.make_tensor_proto(image2, dtype=types_pb2.DT_FLOAT, shape=[1] + list(image2.shape)))
request.inputs['inputs'].CopyFrom(
         tf.contrib.util.make_tensor_proto(image2, shape=[1] + list(image2.shape)))
	
# Call the prediction server
result = stub.Predict(request, 10.0)  # 10 secs timeout

# Plot boxes on the input image
#category_index = load_label_map(FLAGS.path_to_labels)
category_index = label_map_util.load_labelmap(FLAGS.path_to_labels)
print(type(category_index))
#print(category_index)
inspect.getmembers(category_index, predicate=inspect.ismethod)
method_list = [func for func in dir(category_index) if callable(getattr(category_index, func))]
print(method_list)
#print(category_index.__sizeof__)
print("len keys=",len(category_index.item))#91
category_index = label_map_util.create_category_index_from_labelmap(FLAGS.path_to_labels)
boxes = result.outputs['detection_boxes'].float_val
classes = result.outputs['detection_classes'].float_val
print("len classes=", len(classes))#100
scores = result.outputs['detection_scores'].float_val
print("len scores=", len(scores))
print(scores)
image_vis = vis_util.visualize_boxes_and_labels_on_image_array(
#    FLAGS.input_image,
    image2,
    np.reshape(boxes,[100,4]),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8)

# Save inference to disk
#scipy.misc.imsave('%s.jpg'%(FLAGS.input_image), image_vis)
#https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.misc.imsave.html#scipy.misc.imsave
#scipy.misc.imsave(*args, **kwds)
#imsave is deprecated! imsave is deprecated in SciPy 1.0.0, and will be removed in 1.2.0. Use imageio.imwrite instead.
print(type(image_vis))
print(image_vis)
imageio.imwrite('coco-clock-result.png', image_vis)
