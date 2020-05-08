# Tensorflow Examples

Repo to store all tensorflow examples.

# Requirements

matplotlib-3.0.2

```python
pip uninstall matplotlib 
pip install matplotlib 
```

tensorflow-1.12

```python
pip install --upgrade tensorflow
```

# Windows

Install anaconda3  
Create workspace  
Install tensorflow  

```cmd
open anaconda prompt
conda create -n mlspace python=3.5 anaconda
conda activate mlspace
conda install -n mlspace -c conda-forge tensorflow
```
Run `tensorflow_self_check.py` to check installation.

## Troubleshooting 

### Problem
```
ERROR: Failed to import the TensorFlow module.
```
Solution  
create anaconda environment, activate environment, install tensorflow.  
Use `conda list` to check for tensorflow module.  

### Problem
```
TensorFlow successfully installed.
The installed version of TensorFlow does not include GPU support.
```
Solution  
create different environment for gpu support, activate environment, install tensorflow-gpu.
```
open anaconda prompt
conda create -n mlspacegpu
conda activate mlspacegpu
conda install -n mlspacegpu python
pip install tensorflow-gpu
pip install --ignore-installed --upgrade tensorflow-gpu
```
Use `conda list` to check for tensorflow-gpu module.  
```
tensorboard               2.2.1                    pypi_0    pypi
tensorboard-plugin-wit    1.6.0.post3              pypi_0    pypi
tensorflow-gpu            2.2.0                    pypi_0    pypi
tensorflow-gpu-estimator  2.2.0                    pypi_0    pypi
```
Run `tensorflow_test.py gpu 1000` to test installation.  

### Problem
```
2020-05-08 09:34:54.458547: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-05-08 09:34:54.463184: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
```
Solution  
tensorflow-gpu modules and cuda libraries are updated constantly, you'll get this warning.  
Just make sure you have installed cuda corrently and your `PATH` environment variable contains the location of the dll.
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
```
in my case the version installed is `cudart64_100.dll`.  

### Problem
```
2020-05-08 09:43:45.925986: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties:
pciBusID: 0000:01:00.0 name: GeForce GTX 1650 computeCapability: 7.5
coreClock: 1.74GHz coreCount: 14 deviceMemorySize: 4.00GiB deviceMemoryBandwidth: 119.24GiB/s
2020-05-08 09:43:45.934640: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-05-08 09:43:45.940137: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cublas64_10.dll'; dlerror: cublas64_10.dll not found
2020-05-08 09:43:45.948258: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cufft64_10.dll'; dlerror: cufft64_10.dll not found
2020-05-08 09:43:45.953019: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'curand64_10.dll'; dlerror: curand64_10.dll not found
2020-05-08 09:43:45.958048: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cusolver64_10.dll'; dlerror: cusolver64_10.dll not found
2020-05-08 09:43:45.963572: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cusparse64_10.dll'; dlerror: cusparse64_10.dll not found
2020-05-08 09:43:45.967769: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
2020-05-08 09:43:45.972048: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1598] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
```
Solution
```
conda deactivate
conda-env remove -n mlspacegpu
conda create -n mlspacegpu python=3.5
conda activate mlspacegpu
pip install tensorflow-gpu==1.13.1
```
Use `conda list` to check for tensorflow-gpu module.  
```
tensorboard               1.13.1                   pypi_0    pypi
tensorflow-estimator      1.13.0                   pypi_0    pypi
tensorflow-gpu            1.13.1                   pypi_0    pypi
```
Run `tensorflow_test.py gpu 1000` to test installation.  
```
Device: /gpu:0
Time taken: 0:00:01.626970
```

### Problem 
```
Data Science libraries jupyter and notebook are not installed in interpreter Python 3.5.5 64-bit ('mlspacegpu':conda)
```
Solution
```
pip install jupyter
pip install notebook
```
conda list
```
jupyter                   1.0.0                    pypi_0    pypi
jupyter-client            6.1.3                    pypi_0    pypi
jupyter-console           6.1.0                    pypi_0    pypi
jupyter-core              4.6.3                    pypi_0    pypi
...
notebook                  6.0.3                    pypi_0    pypi
```

### Problem
Error when executing `keras-digits-training.ipynb`
```
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
 in 
----> 1 import matplotlib.pyplot as plt
      2 # plot 4 images as gray scale
      3 plt.subplot(221)
      4 plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
      5 plt.subplot(222)

ImportError: No module named 'matplotlib'
```
Solution
```
pip install matplotlib
conda list
matplotlib                3.0.3                    pypi_0    pypi
```

### Problem
```
Failed to start a session for the Kernel 'Python 3': View Jupyter log for further details.
```
Solution  
Click on the button `Select a different Kernel` and choose the `mlspacegpu` environment.  
VSCode -> Output -> Jupyter 
```
[I 10:33:06.914 NotebookApp] Creating new notebook in /
[I 10:33:06.961 NotebookApp] Kernel started: 2b163d74-61d5-44f8-8e76-5c52dbcbe481
```
Also make sure when opening ipynb that top right corner displays Jupyter Server connected and using the right environment.

### Problem
```
 from ._conv import register_converters as _register_converters
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
 in ()
     18 #model.load_weights('keras_digits_model.h5')
     19 # Recreate the exact same model, including weights and optimizer.
---> 20 model = tf.keras.models.load_model('keras_digits_model.h5')
```
Solution  
Use the 'keras_digits_model.h5' provided, don't run 'keras-digits-training.ipynb' 
