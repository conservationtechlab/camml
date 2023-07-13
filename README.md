# camml
A package of ML components for CTL field camera systems. 

# Training custom TFLite models

The `obj_det_train.py` file uses transfer learning to retrain an EfficientDet-Lite object detection model using a given set of images and their corresponding CSV file containing bounded box image data. This model is exported as a TFLite model so it can then be compiled and deployed for a Coral Edge TPU. The steps for this process are as follows:

## Install packages

`sudo apt -y install libportaudio2`
`pip install -q --use-deprecated=legacy-resolver tflite-model-maker`  
`pip install -q pycocotools`  
`pip install -q opencv-python-headless==4.1.2.30`  
`pip uninstall -y tensorflow && pip install -q tensorflow==2.8.0`  

## Steps for training model

1. Run megadetector on a desired set of images. This will produce an output JSON file which contains the bounded box image data.
2. Use the `megadetector_json_to_csv.py` script on your megadetector output JSON file to produce a CSV file in the appropriate format for training the object detector.
3. Use the `obj_det_train.py` script to train the model and export to TFLite.

## Compile the TFLite model for the Edge TPU

`curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -`  
`echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list`  
`sudo apt-get update`  
`sudo apt-get install edgetpu-compiler`  
Pycoral should be installed as well.

## Test the model on Coral Edge TPU

A labels.txt file will need to be made containing:
```
1 animal
2 person
3 vehicle
```

`python3 detect_image.py \  
  --model model_edgetpu.tflite \  
  --labels labels.txt \  
  --input animal.jpg \  
  --output animal_result.jpg`  

# Dependencies

Depends on Google's pycoral package. Instructions for installing this
at:

https://coral.ai/docs/accelerator/get-started/#pycoral-on-linux