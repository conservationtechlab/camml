# camml
A package of ML components for CTL field camera systems. 

## Dependencies

Depends on Google's pycoral package. Instructions for installing this
at:

https://coral.ai/docs/accelerator/get-started/#pycoral-on-linux

# Training models for use with camml

Included in this repository (but currently exposed through the
package) are tools for training models for use with camml.  Camml has
support for YOLOv3 and MobilenetSSD but the training pipelines for
those were not included in this repository.  What is included here are
training tools for YOLOV8 and EfficientDet.  For either, one can train
from manually boxed data or from data that has merely been labeled for
image classification.  In the latter case, MegaDetector is used to
generate the training boxes (with the presumption that all the classes
of interest are animals), applying the image-level labels as the
labels to the boxes.  That latter automated boxing process comes, of
course, with assumptions and risks that should be taken into account.

## Training custom TFLite models

The `obj_det_train.py` file uses transfer learning to retrain an EfficientDet-Lite object detection model using a given set of images and their corresponding CSV file containing bounded box image data. This model is exported as a TFLite model so it can then be compiled and deployed for a Coral Edge TPU. The steps for this process are as follows:

### Setting up virtualenvwrapper environments
Following the training steps may require up to 3 virtual environments to keep each process working. To setup virtualenvwrapper:  
```
sudo pip install virtualenvwrapper
```  
Then edit your ~/.bashrc file with a text editor such as emacs using:  
```
emacs ~/.bashrc
```  
At the end of the file add the following lines:  
```
export WORKON_HOME=~/Envs  
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3  
source /usr/local/bin/virtualenvwrapper.sh  
```  
Then run the script with:  
`source ~/.bashrc`  
This will reset your terminal and you can now use the commands:  
```
mkdir -p $WORKON_HOME  
echo $WORKON_HOME  
source /usr/local/bin/virtualenvwrapper.sh  
```  
You should now be able to create new virtual environments, for example a "camml_training" environment using:  
`mkvirtualenv camml_training`  
This will allow you to quickly activate or switch to environments with:  
`workon camml_training`  
And deactivate the environment with:  
`deactivate`  
Make sure your environment is deactivated before you set up your Anaconda environments.  

### Setting up Anaconda environments
Following the training steps will require at least one Anaconda environment for using Megadetector. To setup Anaconda we will use the same steps described [here](https://docs.anaconda.com/free/anaconda/install/linux/). First open your browser and download the Anaconda installer for Linux [here](https://www.anaconda.com/download/#linux). Open your terminal and enter:  
`bash ~/Downloads/Anaconda3-2023.03-1-Linux-x86_64.sh`  
Where "Downloads" is replaced by your path to the file and the `.sh` filename matches the one you downloaded. Next, press Enter to review the license agreement and press and hold Enter to scroll. Enter "yes" to agree to the license agreement. You can then use Enter to accept the default install location or enter a filepath to specify a different install location. The installation will begin which may take a few minutes. When finished you will be prompted to enter "yes" to initialize conda. You can now either close and reopen your terminal or use `source ~/.bashrc` to refresh the terminal. You should see "(base)" at your command line and you should be ready for the MegaDetector setup steps.  
Use the following command to deactivate your conda environment:  
`conda deactivate base`

### Setting up Open Images Downloaderv4
Open Images Downloaderv4 can be used to download sets of images by class, amount, and set type. This is a good tool to get your training and validation images as well as ground truth bounding box annotations. In particular, the conversion scripts will take the true validation annotations rather than try to guess them with MegaDetector. This makes our final evaluation metrics more trustworthy. You may need a new virtual environment for this step. To begin, enter into your command line:  
```
git clone https://github.com/EscVM/OIDv4_ToolKit.git   
cd OIDv4_ToolKit   
pip install -r requirements.txt  
```  
You can use OIDv4 to download images, for example:  
```
python main.py downloader --classes Cat Dog --type_csv train --limit 50  
python main.py downloader --classes Cat Dog --type_csv validation --limit 50  
```  
When prompted enter "Y" to download the missing annotation file. You should now have 50 Cat and 50 Dog training images as well as 50 Cat and 50 Dog validation images in "/OIDv4_ToolKit/OID/Dataset/". Within your Cat and Dog folders will also be a "Label" folder with `.txt` annotation files for each image.

### Setting up MegaDetector

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
cd ~/git/CameraTraps/envs/
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
This will reload your terminal and you should now be able to activate the environment with:  
`conda activate cameratraps-detector`  
You are now able to run MegaDetector on a given set of images with the following code:  
```
cd ..
python detection/run_detector_batch.py /home/user/megadetector/md_v5a.0.0.pt /home/user/image_folder/ /home/user/megadetector/test_output.json --recursive --checkpoint_frequency 10000
```  
This will produce your `test_output.json` output file and you can now proceed with using these detections to train a TFLite model. Make sure to deactivate your environment before moving on to the next steps:  
`conda deactivate`

### Install packages needed for TFLite Training

Install this Debian package (note: what is this a dependency of?)

    sudo apt -y install libportaudio2

Re-activate the virtual environment you had created before (not the
conda one):

    workon camml_training

Install required pip packages:

    pip install --use-deprecated=legacy-resolver tflite-model-maker
    pip install pycocotools
    pip install opencv-python-headless==4.1.2.30
    pip uninstall -y tensorflow && pip install tensorflow==2.8.0

#`pip install humanfriendly` (delete?)

### Steps for training model

1. Run megadetector on a desired set of images with `git/CameraTraps/run_detector_batch.py`. If you've completed the "Setting up MegaDetector steps you should be ready to go. This will produce an output `.json` file which contains the bounded box image data. The detections are made in 3 classes 1-animal, 2-person, and 3-vehicle.  
Important Note: You can run MegaDetector on either all your images at once or only on your training images. If you choose all your images at once they will be randomly sorted between train, validation, and test with a roughly 80/10/10 split and all detections will be filtered by the same confidence value you provide. You should then use the `megadetector_json_to_csv.py` script for the next step. If you choose to run MegaDetector only on your training images you should use the `md_json_to_csv_valtest.py` script which will convert your MegaDetector output `.json` file as normal but then convert the validation and test OIDv4 annotations into the same format. This second method produces more trustworthy mean Average Precision metrics.  
2. Use the `megadetector_json_to_csv.py` (or `md_json_to_csv_valtest.py`) script on your megadetector output `.json` file to produce a `.csv` file in the appropriate format for training the object detector. The 'person' and 'vehicle' detections will be excluded and the 'animal' class will be changed to the folder name where the images came from. For example: Any 'animal' detections on images from the directory /home/usr/images/Cat/ will change to 'Cat' detections in the CSV file. Since megadetector's detections each come with their own confidence score you can set the confidence argument to a value between 0 and 1, such as 0.9, to only include detections which have a confidence greater than or equal to this value. To run the script enter:    
```
mkdir ~/tflite_train
python megadetector_json_to_csv.py ~/megadetector/test_output.json ~/tflite_train/test_output.csv 0.9
```
  
3. Use the `obj_det_train.py` script to train the model and export to TFLite. You should now see a `model.tflite` file in your current directory. To run the script enter:
```
python obj_det_train.py ~/tflite/test_output.csv
```
  
You can now deactivate the virtual environment you used for training:

    deactivate
  
### Compile the TFLite model for the Edge TPU
You may need a new virtual environment to compile this model for the Coral.  
`curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -`  
```
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
```  
`sudo apt-get update`  
`sudo apt-get install edgetpu-compiler`  
  
`sudo apt-get install libedgetpu1-std`  
`sudo apt-get install python3-pycoral`  
`git clone https://github.com/google-coral/pycoral.git`  

Compile model to run on  1 Edge TPU:
`edgetpu_compiler model.tflite --num_segments=1`

### Test the model on Coral Edge TPU

A labels.txt file will need to be made containing class names such as:
```
0 Cat
1 Dog
```  

Run the Edge compiled TFLite model on the Coral:  
```
python3 /pycoral/examples/detect_image.py --model model_edgetpu.tflite --labels labels.txt --input animal.jpg --output animal_result.jpg
```  

If this step fails, you may also need to install the TFLite runtime associated with your Linux OS and Python versions.  

## Training custom YOLOv8 models
Yolov8 models use individual PyTorch `.txt` annotation files which are each associated with an image. These annotation files should be in a folder at the same level as the images folder such as "/home/user/project/images/" and "/home/user/project/labels/". You will need a `.yaml` file that contains the path to the train, validation, and test directories as well as the number of classes in your dataset and the class names. The conversion script `megadetector_json_to_pt.py` will use the megadetector output file and the path to your validation images to create a folder structure with symlinked images and `.txt` files in the required format. It will also create the necessary  data.yaml file for training. For example:  
```
train: /home/user/yolov8_training_data/train/images  
val: /home/user/yolov8_training_data/validation/images  
#test: /home/user/yolov8_training_data/test/images	# test set is optional  

nc: 2
names: ['Cat', 'Dog']
```  

### Setting up Open Images Downloaderv4
Follow the "Setting up Open Images Downloaderv4" steps as described in the "Training custom TFLite models" section.  

### Setting up megadetector
Follow the "Setting up megadetector" steps as described in the "Training custom TFLite models" section.  

### Install packages
You should create a new virtual environment before pip installing.  
`pip install ultralytics`  

### (Optional) Install MLops software
If you want to visualize your training data and results you can use ClearML. You will have to create an account and verify your credentials, then any YOLOv8 training experiment ran in this terminal will be logged to your ClearML dashboard.  
1. First create a ClearML account.  
2. Once logged in go to "Settings" then "Workspace" and click on "create new credentials".  
3. Then copy the information under "LOCAL PYTHON".  
4. You can now enter in your terminal:  
`clearml-init`  
Then paste your credential information and your results should be automatically logged once you start training.  

### Steps for training model
1. Run megadetector on a desired set of images with `git/CameraTraps/run_detector_batch.py`. If you've completed the "Setting up MegaDetector steps you should be ready to go. We recommend running megadetector only on your training images to keep your test and validation sets untouched. This will produce an output `.json` file which contains the bounded box image data. The detections are made in 3 classes 1-animal, 2-person, and 3-vehicle.  
Important Note: A training image set and validation image set is required to train a YOLOv8 model, the test image set is optional. You should run MegaDetector only on your training images. If you don't include a test image set you should then use the `md_json_to_pt_val.py` script for the next step. If you choose to include a test image set then you should use the `md_json_to_pt_valtest.py` script which will convert your MegaDetector output `.json` file as normal but then convert the validation and test OIDv4 annotations into the same format. This second method should produce more trustworthy mean Average Precision metrics but results are inconclusive.  

2. Use the `md_json_to_pt_val.py` (or `md_json_to_pt_valtest.py`) script on your megadetector output `.json` file to produce PyTorch `.txt` files in the appropriate format for training the object detector. The script will create a "yolov8_training_data/train/images" and "yolov8_training_data/train/labels" directory structure at your current directory level. Similarly, a "yolov8_training_data/validation/images" and "yolov8_training_data/validation/labels/" will also be created. The megadetector output will be converted and stored in the "train" folder while the given validation data will be converted and stored in the "validation" folder. Symlinks of each image will be created so our Yolo training works without issue. The 'person' and 'vehicle' detections will be excluded and each detection will be given a class number to match the classes in the `data.yaml` file. Since megadetector's detections each come with their own confidence score you can set the confidence argument to a value between 0 and 1, such as 0.9, to only include detections which have a confidence greater than or equal to this value. To run the script enter:    
```
python megadetector_json_to_pt.py test_output.json /home/user/project/validation/  0.9
```
  
3. A YOLOv8 model file will automatically be downloaded when you begin training, or you can download the model file yourself [here](https://docs.ultralytics.com/models/yolov8/#supported-tasks). These instructions will use the `yolov8n.pt` model. Your model file and `data.yaml` file should both be in your current working directory.  

Enter in your terminal:  
```
yolo task=detect mode=train model=yolov8n.pt imgsz=1280 data=data.yaml epochs=50 batch=8 name=yolov8n_50e
```  
We use "task=detect" and "mode=train" to train an object detector. We set "model=yolov8n.pt" which is the smallest and probably least accurate model but you can use any model version. Standard image size is 640 but training with "imgsz=1280" can give better results on images with many small objects that you want to classify. The "data.yaml" file should be made as described above. Epochs and batch size can affect training time and AP results. For best practices see [here](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/#training-settings). Training may take a few hours or longer depending on your hardware.  
4. When training is complete your results should be saved in "runs/detect/yolov8n_50/" and your final trained model will be at "runs/detect/yolov8n_50e/weights/best.pt".

### Test the model
To test the model download a class image from google and use `test_yolo_dnn.py` file with your model filename and image filename to convert the `.pt` model file to a `.onnx` model file and run inference on the image. For example:

