import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

### https://www.tensorflow.org/guide/gpu#using_a_single_gpu_on_a_multi-gpu_system
### matrix multiplication
### argv[1] = cpu or gpu 
### argv[2] = size of matrix

device_name = sys.argv[1]
thematrix = (int(sys.argv[2]), int(sys.argv[2]))
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

with tf.device(device_name):
    random_matrix = tf.random.uniform(shape=thematrix, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)

startTime = datetime.now()
with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as session:
        with tf.device(device_name):
            random_matrix = tf.random.uniform(shape=thematrix, minval=0, maxval=1)
            dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
            sum_operation = tf.reduce_sum(dot_operation)

        result = session.run(sum_operation)
        print(result)

### Print the result 
print("Device:", device_name)
print("Time taken:", datetime.now() - startTime)
