# camml
A package of ML components for CTL field camera systems. 

# Training custom TFLite models

The `obj_det_train.py` file uses transfer learning to retrain an EfficientDet-Lite object detection model using a given set of images and their corresponding CSV file containing bounded box image data. This model is exported as a TFLite model so it can then be compiled and deployed for a Coral Edge TPU. The steps for this process are as follows:

## Setting up MegaDetector

The steps for setting up and using Megadetector are described [here](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md). As the instructions can be subject to change we will also guide you through the instructions ourselves, as they were presented to us at the time. All instructions will assume you're using a Linux operating system.  

To setup and use MegaDetector you'll need [Anaconda](https://www.anaconda.com/products/individual) and Git installed. Anaconda is used to create self contained environment where you can pip install specific packages and dependencies without the worry of creating conflicts and breaking other setups. You may also need to have recent NVIDIA drivers if you plan to use a GPU to speed up detection.  

Next you'll need to download a MegaDetector model file such as [MDv5a](https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt). The instructions recommend putting this model file in a "megadetector" folder in your home directory, such as "/home/user/megadetector/", to match the rest of the instructions.  

With Anaconda installed you should see "(base)" at your command prompt. Run the following instructions to clone the required github repositories and create the anaconda environment to run MegaDetector:
```
mkdir ~/git
cd ~/git
git clone https://github.com/ecologize/yolov5/
git clone https://github.com/ecologize/CameraTraps
git clone https://github.com/Microsoft/ai4eutils
cd ~/git/cameratraps
conda env create --file environment-detector.yml
conda activate cameratraps-detector
export PYTHONPATH="$PYTHONPATH:$HOME/git/CameraTraps:$HOME/git/ai4eutils:$HOME/git/yolov5"
```
You should now see "(cameratraps-detector)" instead of "(base)" at your command prompt. Now we'll add the line to update our PYTHONPATH to our .bashrc file. Open your .bashrc file in an editor, such as emacs.
`emacs ~/.bashrc`
Add this line to the end of your .bashrc file:
`export PYTHONPATH="$PYTHONPATH:$HOME/git/CameraTraps:$HOME/git/ai4eutils:$HOME/git/yolov5"`
Then save, exit and run:
`source ~/.bashrc`
This will reload your terminal and you should now be able to activate the enviroment with:
`conda activate cameratraps-detector`
You are now be able to run MegaDetector on a given set of images with the following code:
`python detection/run_detector_batch.py /home/user/megadetector/md_v5a.0.0.pt /home/user/image_folder/ /home/user/megadetector/test_output.json --output_relative_filenames --recursive --checkpoint_frequency 10000`
This will produce your `test_output.json` output file and you can now proceed with using these detections to train a TFLite model. Make sure to deactivate your environment before moving on to the next steps:
`conda deactivate`

## Install packages

`sudo apt -y install libportaudio2`  
We'll want to create a new virtual environment to download the packages needed for training.
`pip install --use-deprecated=legacy-resolver tflite-model-maker`  
`pip install pycocotools`  
`pip install opencv-python-headless==4.1.2.30`  
`pip uninstall -y tensorflow && pip install tensorflow==2.8.0`
`pip install humanfriendly`

## Steps for training model

1. Run megadetector on a desired set of images with `git/CameraTraps/run_detector_batch.py`. If you've completed the "Setting up MegaDetector steps you should be ready to go. This will produce an output `.json` file which contains the bounded box image data. The detections are made in 3 classes 1-animal, 2-person, and 3-vehicle.  
2. Use the `megadetector_json_to_csv.py` script on your megadetector output `.json` file to produce a `.csv` file in the appropriate format for training the object detector. The 'person' and 'vehicle' detections will be excluded and the 'animal' class will be changed to the folder name where the images came from. For example: Any 'animal' detections on images from the directory /home/usr/images/Cat/ will change to 'Cat' detections in the CSV file. Since megadetector's detections each come with their own confidence score you can set the confidence argument to a value between 0 and 1, such as 0.9, to only include detections which have a confidence greater than or equal to this value.  
3. Use the `obj_det_train.py` script to train the model and export to TFLite. You should now see a `model.tflite` file in your current directory.

## Compile the TFLite model for the Edge TPU
You may need a new virtual environment to compile this model for the Coral.  
`curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -`  
`echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list`  
`sudo apt-get update`  
`sudo apt-get install edgetpu-compiler`  
  
`sudo apt-get install libedgetpu1-std`  
`sudo apt-get install python3-pycoral`  
`git clone https://github.com/google-coral/pycoral.git`  

Compile model to run on  1 Edge TPU:
`edgetpu_compiler model.tflite --num_segments=1`

## Test the model on Coral Edge TPU

A labels.txt file will need to be made containing class names such as:
```
0 Cat
1 Dog
```  

Run the Edge compiled TFLite model on the Coral:  
`python3 /pycoral/examples/detect_image.py --model model_edgetpu.tflite --labels labels.txt --input animal.jpg --output animal_result.jpg`  

If this step fails, you may also need to install the TFLite runtime associated with your Linux OS and Python versions.

# Dependencies

Depends on Google's pycoral package. Instructions for installing this
at:

https://coral.ai/docs/accelerator/get-started/#pycoral-on-linux