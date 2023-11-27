# camml
A package of ML components for CTL field camera systems.

The package is a high-level library for doing image classification and
object detection on edge camera devices particularly targeted towards
dependent systems developed and used by the Conservation Technology
Lab and its partners.

At this stage, there are two facets of this repository: 1) the actual
camml package whose releases are served at PyPI and 2) a training
pipeline to create custom object detector models for use with the
camml package that is only accessible through the repo itself. At this
moment, almost all the remainder of this README (beyond the next
subsection) is concerned with the training facet.

## Dependencies of the package

The camml package installs most of its dependencies automatically
except that it also depends on Google's pycoral package which is not
served by PyPI. Instructions for installing this are at:

https://coral.ai/docs/accelerator/get-started/#pycoral-on-linux

# Training models for use with camml

Included in this repository (but not currently exposed through the
package) are tools for training models for use with camml.  Camml has
support for YOLOv3 and MobilenetSSD models but the training pipelines
for those were not included in this repository.  What is included here
are training tools for YOLOv8 and EfficientDet.  For either, one can
train from manually boxed data or from data that has merely been
labeled for image classification.  In the latter case, MegaDetector is
used to generate the training boxes (with the presumption that all the
classes of interest are animals), applying the image-level labels as
the labels to the boxes.  That latter automated boxing process comes,
of course, with assumptions and risks that should be taken into
account.

The first step is to clone this repository:

    mkdir ~/git
    cd ~/git
    git clone https://github.com/conservationtechlab/camml.git

## Setting up virtualenv environments

As always, it is advisable to keep your python environments for
specific tasks (like this training process) separate from your main
system's python environment.  Unfortunately, as things stand,
different parts of this training process require different
configurations of python packages so you'll need multiple environments
that you have to activate and deactivate at different stages of the
process.  We'll use `virtualenv` for at least one of these virtual
environments. Here's how to get that set up:

Install virtualenvwrapper:  

    sudo pip install virtualenvwrapper

Edit your ~/.bashrc file with a text editor such as emacs:  

    emacs ~/.bashrc
  
At the end of that file add the following lines:  

    export WORKON_HOME=~/envs  
    export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3  
    source /usr/local/bin/virtualenvwrapper.sh  

Run the script with:  

    source ~/.bashrc

Create the folder you've designated for the virtual environments:

    mkdir -p $WORKON_HOME  

You should now be able to create new virtual environments.  In this
README, we'll use one called `camml_tflite_training` for the
EfficientDet TFLITE training pipeline and one called
`camml_yolov8_training` for the YOLOv8 training pipeline and we will
create them thusly:

    mkvirtualenv camml_tflite_training
    mkvirtualenv camml_yolov8_training

Each environment will be activated after you created it. When you want
to deactivate a given virtualenv environment:

    deactivate

And to re-activate it when you need it again:

    workon camml_tflite_training

Make sure your environment is deactivated before you set up or activate your
Anaconda environments for Megadetector later in the README.


## Downloading example image data

When getting started with training custom object detector models for
camml, it can be useful to download existing, publicly available
labeled data.  OpenImages is a good source of such data and the
downloader tool installed in this section provides an easy way to
download sets of images by class, amount, and split type from
OpenImages v4; to get training and validation images for use during
training in addition to test images with ground truth bounding box
annotations for use in evaluating whether training with automated
boxing by MegaDetector produced acceptable performance. 

Clone OpenImagesv4 downloader tool and install its dependencies:	

    cd ~/git           
    git clone https://github.com/EscVM/OIDv4_ToolKit.git
    cd OIDv4_ToolKit
    workon camml_tflite_training
    pip install -r requirements.txt

Run tool. To download 50 images each of cats and dogs from the
training and validation sets on OpenImages:

    python main.py downloader --classes Cat Dog --type_csv train --limit 50
    python main.py downloader --classes Cat Dog --type_csv validation --limit 50

When prompted enter "Y" to download the missing annotation file. You
should now have 50 Cat and 50 Dog training images as well as 50 Cat
and 50 Dog validation images in
"~/git/OIDv4_ToolKit/OID/Dataset/". Within the Cat and Dog folders
will also be a "Label" folder with `.txt` annotation files for each
image.

## Setting up an Anaconda environment for Megadetector

Both the TFLite and YOLOv8 training pipelines described below leverage
an off-the-shelf general-animal object detector called Megadetector
that is commonly used in the camera trap community.  In this project,
we use Megadetector to automatically create boxed data from data only
labelled at the image level to produce data that can be used to train
custom object detectors (user beware: this strategy has assumptions
and possible pitfalls).

We will need an Anaconda environment for using Megadetector. To setup
Anaconda we will use the same steps described
[here](https://docs.anaconda.com/free/anaconda/install/linux/). These
steps are:

Open a web browser and download the Anaconda installer for Linux
[here](https://www.anaconda.com/download/#linux).

To be safe, at this stage, make sure to deactivate any virtualenv
environments you have activated:

    deactivate

In a terminal, navigate to where you downloaded the Anaconda installer
script and run (replacing the name of the Anaconda install script to
match the one you just downloaded):

    bash Anaconda3-2023.03-1-Linux-x86_64.sh

Press enter to review the license agreement and press and hold enter
to scroll (or hit 'q' to jump forward if you are feeling
trusting). Enter "yes" to agree to the license agreement. Hit enter to
accept the default install location or enter a filepath to specify a
different install location. The installation will begin and may take a
few minutes. When finished you will be prompted to enter "yes" to
modify your shell profile to automatically initialize conda. You can
now either close and reopen your terminal or use `source ~/.bashrc` to
refresh the terminal. You should see "(base)" at your command line and
you should be ready for the MegaDetector setup steps.

To deactivate the conda environment:

    conda deactivate base`

## Setting up MegaDetector

The steps for setting up and using Megadetector are described
[here](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md). As
the instructions can be subject to change we will also guide you
through the instructions ourselves, as they were presented to us at
the time. All instructions will assume you're using a Linux operating
system.

To set up and use MegaDetector you'll need
[Anaconda](https://www.anaconda.com/products/individual). Anaconda
(installation instructions above) is used to create self contained
environment where you can install specific packages and dependencies
without the worry of creating conflicts and breaking other setups. You
may also need to have recent NVIDIA drivers if you plan to use a GPU
to speed up detection.

Download a MegaDetector model file. For example, the following command
will download MegaDetector v5a:

    mkdir ~/megadetector
    cd ~/megadetector
    wget https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt

These commands will put the model file in a "megadetector" folder in
your home directory, such as `/home/<user>/megadetector/`, which
matches the rest of their and our instructions.

With Anaconda installed per instructions above, you should see
`(base)` at your command prompt. Run the following instructions to
clone the required github repositories and create the anaconda
environment in which you will run MegaDetector:

    cd ~/git
    git clone https://github.com/ecologize/yolov5/
    git clone https://github.com/ecologize/CameraTraps
    git clone https://github.com/Microsoft/ai4eutils
    cd ~/git/CameraTraps/envs/
    conda env create --file environment-detector.yml
    conda activate cameratraps-detector

You should now see `(cameratraps-detector)` instead of `(base)` at
your command prompt.

Set the PYTHONPATH: 

    export PYTHONPATH="$PYTHONPATH:$HOME/git/CameraTraps:$HOME/git/ai4eutils:$HOME/git/yolov5"

If you want PYTHONPATH to always automatically be set, add the above
line to your .bashrc file. To do this, open your .bashrc file in an
editor, such as emacs:

    emacs ~/.bashrc
    
And paste this line at the end of the file:  

    export PYTHONPATH="$PYTHONPATH:$HOME/git/CameraTraps:$HOME/git/ai4eutils:$HOME/git/yolov5"

Save the file, exit and run:  

    source ~/.bashrc
    
to reload your settings.

When you want to deactivate the conda environment:

    conda deactivate

To re-activate when you need it:

    conda activate cameratraps-detector`

as you will in the next section.

## Using MegaDetector to automatically generate boxed data

With MegaDetector installed, you are now able to run MegaDetector on a
given set of images:

    conda activate cameratraps-detector
    cd ~/git/CameraTraps
    python detection/run_detector_batch.py ~/megadetector/md_v5a.0.0.pt ~/git/OIDv4_ToolKit/OID/Dataset/train/ ~/megadetector/demo.json --recursive --checkpoint_frequency 10000

Here, we have used the training images that were downloaded from OIDv4
in the earlier part of this README as an example. If you were to use
other data, you'd need to change the second argument of the last
command to point the tool those images.

This will produce a `demo.json` output file and you can now
proceed with using these detections to train a TFLite model. Make sure
to deactivate your environment before moving on to the next steps:

    conda deactivate

And, actually, since these instructions said to allow conda to
automatically load into the base conda environment you may need to
deactivate from that, too (won't hurt anything to issue this command
even if not currently in a conda environment):

    conda deactivate

## Training custom TFLite models

The `obj_det_train.py` script uses transfer learning to retrain an
EfficientDet-Lite object detection model using a given set of images
and a corresponding CSV file containing bounded box image data that is
in turn generated from the MegaDetector JSON file created above. This
model is exported as a TFLite model so it can then be compiled and
deployed for a Coral Edge TPU, which is one of the primary targets
used in `camml`. The steps for this process are as follows:

### Install packages needed for TFLite Training

Install this Debian package (note: what is this a dependency of?)

    sudo apt -y install libportaudio2

Re-activate the virtual environment you had created before (not the
conda one):

    workon camml_tflite_training

Install required pip packages:

    pip install --use-deprecated=legacy-resolver tflite-model-maker
    pip install pycocotools
    pip install opencv-python-headless==4.1.2.30
    pip uninstall -y tensorflow && pip install tensorflow==2.8.0
    pip install numpy==1.23.5

This may also be necessary (need to check):

    pip install protobuf==3.20.3

NOTE: we are installing lots of older versions of packages because at
the time of the creation of this README that was what had been
determined to be necessary for the core package, `tflite-model-maker`
to function.

### Steps for training model

NOTE: There may be some assumptons in the below process that you are
using the training images that were downloaded as example data from
OIDv4 in the earlier part of this README.  If you were to use other
data, you would likely have to do some work to prepare it to match the
format of that data. Instructions for doing this will ideally be added
to this README later.

#### 1. Run MegaDetector on the data

Run megadetector on a desired set of images with
`git/CameraTraps/run_detector_batch.py` (see instructions in earlier
section of this README. If you've completed that section, you should
be ready to go and can skip this step). This will produce an output
`.json` file which contains the bounded box image data.

Important Note: You can run MegaDetector on either all your images at
once or only on your training images. If you choose all your images at
once they will be randomly sorted between train, validation, and test
with a roughly 80/10/10 split and all detections will be filtered by
the same confidence value you provide. You should then use the
`megadetector_json_to_csv.py` script for the next step. If you choose
to run MegaDetector only on your training images you should use the
`md_json_to_csv_valtest.py` script which will convert your
MegaDetector output `.json` file as normal but then convert the
validation and test OIDv4 annotations into the same format (NOTE: we
might want to look into whether we want to grab the OID boxes for
validation as well as test if this is indeed what is happening (is
what the README just said).  The plan had been to only grab test
boxes from OID.  Probably hasn't affected things too much to have val
boxes come from manual labels to this stage as I don't believe folks
were tuning training based on validation but once we are, might want
to have a think about what it means to do so and whether it is better
or worse than the original plan). This second method produces more
trustworthy mean Average Precision metrics.

#### 2. Generate CSV for training from MegaDetector JSON

Use the `megadetector_json_to_csv.py` (or `md_json_to_csv_valtest.py`)
script on your megadetector output `.json` file to produce a `.csv`
file in the appropriate format for training the object
detector. MegaDetector detections are one of 3 classes, with a number
representing each class: 1 for animal, 2 for person, and 3 for
vehicle.  The 'person' and 'vehicle' detections will be excluded by
the CSV preparation scripts and the 'animal' class will be changed to
the folder name where the images came from. For example: Any 'animal'
detections on images from the directory /home/usr/images/Cat/ will
change to 'Cat' detections in the CSV file. Since Megadetector's
detections each come with their own confidence score you can set the
confidence argument to a value between 0 and 1, such as 0.15, to only
include detections which have a confidence greater than or equal to
this value.

To run the script enter:

    cd ~/git/camml/training/
    mkdir ~/tflite_train
    python megadetector_json_to_csv.py ~/megadetector/demo.json ~/tflite_train/demo.csv 0.15


#### 3. Perform training using CSV as input

Use the `obj_det_train.py` script to train the model and export a
TFLite model:

    python obj_det_train.py ~/tflite_train/demo.csv ~/tflite_train/demo -m demo.tflite

NOTE: Known issue in some installs when running previous line has been that
TensorFlow needs permissions on /tmp and doesn't have them.  The fix
has been to make sure Tensorflow can write its cache by storing it in
user's own space:

    export TFHUB_CACHE_DIR=./tmp

If `obj_det_train.py` script did run successfully, you should now see
a `model.tflite` file in your current directory.
  
You can now deactivate the virtual environment you used for training:

    deactivate
  
### Compile the TFLite model for the Edge TPU

Install necessary packages:

    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    sudo apt update
    sudo apt install edgetpu-compiler

Compile model to run on one Edge TPU:

    cd ~/tflite_train/demo/
    edgetpu_compiler demo.tflite --num_segments=1

This will create a file called model_edgetpu.tflite sitting in active
directory.

Deactivate the virtual environment you were working in:

    deactivate

### Test the trained model on Coral Edge TPU

These instructions are specifically for testing the model on a Coral
USB Accelerator device (as opposed to another Coral TPU-enabled
device/product).

Deactivate all virtual environments you were working with. Install
these packages:

    sudo apt install libedgetpu1-std
    sudo apt install pycoral-examples

At this point, you should physically connect the USB Coral Accelerator
to the computer.

A label map file will need to be created containing class names such
as: ``` 0 Cat 1 Dog ```

You can extract this label map from the non-edgetpu tflite model by
unzipping it:

    unzip demo.tflite

This produces a file called `labelmap.txt` with each class label on
its own line (the class labels are in order so that the line number
corresponds with the class index associated with the class label)

Run the Edge compiled TFLite model on the Coral:  

    python3 /usr/share/pycoral/examples/detect_image.py --model demo_edgetpu.tflite --labels labelmap.txt --input animal.jpg --output animal_result.jpg

Where animal.jpg is any animal photo you have lying around or perhaps
one from the test data set.  You could use an image from the training
set just to see that the script runs without issue but be wary that
the model will perform particularly/misleadingly well on data it was
trained with (for clear reasons).

NOTE: if using current `pillow` package (as of 2023-11-27) the script
will fail due to a change in `pillow` that is not reflected in the
`detect_image.py`. Here's a hacky fix:

   sudo sed -i 's/ANTIALIAS/LANCZOS/g' /usr/share/pycoral/examples/detect_image.py

If this `detect_image.py` script still fails, you may also need to
install the TFLite runtime associated with your Linux OS and Python
versions.

## Training custom YOLOv8 models

Yolov8 models use individual PyTorch `.txt` annotation files which are each associated with an image. These annotation files should be in a folder at the same level as the images folder such as "/home/user/project/images/" and "/home/user/project/labels/". You will need a `.yaml` file that contains the path to the train, validation, and test directories as well as the number of classes in your dataset and the class names. The conversion script `megadetector_json_to_pt.py` will use the megadetector output file and the path to your validation images to create a folder structure with symlinked images and `.txt` files in the required format. It will also create the necessary  data.yaml file for training. For example:  
```
train: /home/user/yolov8_training_data/train/images  
val: /home/user/yolov8_training_data/validation/images  
#test: /home/user/yolov8_training_data/test/images	# test set is optional  

nc: 2
names: ['Cat', 'Dog']
```  

### Download some example data

To try out the pipeline with an example dataset, follow the steps for
installing the Open Images v4 data downloader as described in an
earlier section.

### Setting up megadetector
Follow the "Setting up megadetector" steps as described in the "Training custom TFLite models" section.  

### Install necessary package dependencies

You should switch to the virtual environment you created before for
this training before installing the pip packages associated with this
training pipeline:

    workon camml_yolov8_training

And then install the packages:

    pip install ultralytics`  

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
    
    python megadetector_json_to_pt.py demo.json /home/user/project/validation/  0.9
  
3. A YOLOv8 model file will automatically be downloaded when you begin training, or you can download the model file yourself [here](https://docs.ultralytics.com/models/yolov8/#supported-tasks). These instructions will use the `yolov8n.pt` model. Your model file and `data.yaml` file should both be in your current working directory.  

Enter in your terminal:  
```
yolo task=detect mode=train model=yolov8n.pt imgsz=1280 data=data.yaml epochs=50 batch=8 name=yolov8n_50e
```  
We use "task=detect" and "mode=train" to train an object detector. We set "model=yolov8n.pt" which is the smallest and probably least accurate model but you can use any model version. Standard image size is 640 but training with "imgsz=1280" can give better results on images with many small objects that you want to classify. The "data.yaml" file should be made as described above. Epochs and batch size can affect training time and AP results. For best practices see [here](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/#training-settings). Training may take a few hours or longer depending on your hardware.  
4. When training is complete your results should be saved in "runs/detect/yolov8n_50/" and your final trained model will be at "runs/detect/yolov8n_50e/weights/best.pt".

### Test the model
To test the model download a class image from google and use `test_yolo_dnn.py` file with your model filename and image filename to convert the `.pt` model file to a `.onnx` model file and run inference on the image. For example:

