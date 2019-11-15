# Tensorflow serving howto 

1. Train model
2. Save model 
3. Serve model 
4. Create client 
5. Execute client 

# Requirements

windows
setup python environment

* install anaconda 
* conda create 
* conda install pip python=3.5
* conda install tensorflow=1.13
* conda install -c anaconda protobuf  

# Train model 

checkout https://github.com/tensorflow/models.git
copy 
export_model.py
exporter.py (overwrite)
to research\object_detection

download ssd_mobilenet_v1_coco_2018_01_28.tar.gz extract to research\object_detection\ssd_mobilenet_v1_coco_2018_01_28
edit ssd_mobilenet_v1_coco_2018_01_28\pipeline.config 
see pipeline.config for example 

generate protos 

```
cd research\object_detection
for /f %i in ('dir /b object_detection\protos\*.proto') do protoc --python_out=. object_detection\protos\%i
```

# Save model 

Downloaded model (ssd_mobilenet_v1_coco_2018_01_28.tar.gz) does not contain variables so have to resave model.  
Edit export_model.py set the paths  

```
cd research\object_detection
python export_model.py
...
...
...
  Preprocessor/map/while/Less (1/1 flops)
  Preprocessor/map/while/Less_1 (1/1 flops)
  Preprocessor/map/while/add (1/1 flops)
  Preprocessor/map/while/add_1 (1/1 flops)

======================End of Report==========================

```

Check model 

```
>saved_model_cli show --dir object_detection/1/saved_model --all
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['detection_signature']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_UINT8
        shape: (-1, -1, -1, 3)
        name: image_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['detection_boxes'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100, 4)
        name: detection_boxes:0
    outputs['detection_classes'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100)
        name: detection_classes:0
    outputs['detection_scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100)
        name: detection_scores:0
    outputs['num_detections'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: num_detections:0
  Method name is: tensorflow/serving/predict

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_UINT8
        shape: (-1, -1, -1, 3)
        name: image_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['detection_boxes'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100, 4)
        name: detection_boxes:0
    outputs['detection_classes'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100)
        name: detection_classes:0
    outputs['detection_scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100)
        name: detection_scores:0
    outputs['num_detections'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: num_detections:0
  Method name is: tensorflow/serving/predict
```

# Running tensorflow serving

Install docker 
Copy saved_model to location where it is reachable by docker 

```
TESTDATA="/home/docker/users/tensorflow-serving/tensorflow_serving/servables/tensorflow/testdata"
echo $TESTDATA
docker run -t --name tensorflow_serving --rm -p 8500:8500 -p 8501:8501 \
    -v "$TESTDATA/object_detection:/models/object_detection" \
    -e MODEL_NAME=object_detection \
    tensorflow/serving:1.13.0 &
...
...
...	
2019-11-14 06:50:22.754045: I tensorflow_serving/servables/tensorflow/saved_model_warmup.cc:101] No warmup data file found at /models/object_detection/1/assets.extra/tf_serving_warmup_requests
2019-11-14 06:50:22.758381: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: object_detection version: 1}
2019-11-14 06:50:22.760190: I tensorflow_serving/model_servers/server.cc:313] Running gRPC ModelServer at 0.0.0.0:8500 ...
[warn] getaddrinfo: address family for nodename not supported
[evhttp_server.cc : 237] RAW: Entering the event loop ...
2019-11-14 06:50:22.761252: I tensorflow_serving/model_servers/server.cc:333] Exporting HTTP/REST API at:localhost:8501 ...

```

# Create client 


# Execute client 


