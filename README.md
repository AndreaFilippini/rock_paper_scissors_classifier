## Real-time classification of hand gestures of "rock, paper, scissors" game
Deep learing project for master's degree in computer and automation engineering.  
This project contains the code to implement a CNN that can classify the hand gestures of "rock, paper, scissors" game in real time.  
The optimal structure was find after 33 epochs, with a validation accuracy of 0.9458 and a validation loss of 0.1364.  
The classification is done using the webcam of the user.  


![alt text](https://github.com/AndreaFilippini/rock_paper_scissors_classifier/blob/main/final_result/result.gif?raw=true)

## Dependencies
[Tensorflow](https://www.tensorflow.org/)  
[Keras](https://keras.io/)  
[OpenCV](https://opencv.org/)  
[CVZone](https://www.computervision.zone/)  
[Jupyter](https://jupyter.org/)  

## Costume dataset
The net was trained on a costum dataset that conatains 36084 images, in which each one has a size of 300x300x3 and a different background.  
The images were splitted in a train set, validation test and test set with a 28764, 5856 and 1464 images respectively.  
The folder structure is as follows:
```
hand-dataset [36084]  
├── rps_train  
│   ├── rock [9588 images]  
│   ├── paper [9588 images]  
│   └── scissors [9588 images]  
├── rps_val
│   ├── rock [488 images]
│   ├── paper [488 images]
│   └── scissors [488 images]
└── rps_test
    ├── rock [1952 images]
    ├── paper [1952 images]
    └── scissors [1952 images]
```
The reason why each images has a different bg is beacause the net is able to generalize better, recognizing the shape of the hand regardless of the background.

## Jupyter & Colab
How connect Jupyter with Colaboratory
### Install Jupyer 
Install with pip:
```
$ pip install jupyter
```
### Install Jupyter server extension for using a WebSoket to proxy HTTP traffic
The colab team authored the Jupyter_http_over_ws extension. Install and enable it by doing:
```
$ pip install jupyter_http_over_ws
$ jupyter serverextension enable --py jupyter_http_over_ws
```
### Start a local Jupyter server
We need a local Jupyter server that trusts WebSocket connections from the Colab frontend. The following command and flags accomplish this:
```
$ jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0
```
Once the server has started, it will print a message with the initial backend URL used for authentication. You’ll need a copy of this in the next step.
### Connect
In Colab, click the “Connect” button and select “Connect to local runtime”. Enter the URL you just copied and click “Connect”.
## TensorFlow
TensorFlow is an open-source machine learning library written in Python and built by Google
### Installing Pre-requisites
Install and check installation Python and pip
```
$ sudo apt install -y python3
$ sudo apt install -y pip
$ python3 -V
$ pip -V
```
### Install Python Virtualenv
We used the venv module creating a virtual environment that is in the python3-venv package. Command used to install the required packages of python3-venv:
```
$ sudo apt install python3-venv python3-dev
```
### Setup virtual enviroment
After installing the venv module, is necessary to set up a virtual environment for the TensorFlow. Navigate into the directory where you want to store the Python 3 virtual environment. You can store it into your home directly or as well in any other directory where you have read and write permissions. Create a directory by using the following mkdir command for the TensorFlow project and the move into it using the cd command as follows:
```
$ mkdir my_tensorflow
$ cd my_tensorflow
```
Using the following command you can create a python virtual environment in the current directory:
```
$ python3 -m venv venv
```
In the above command, the second word venv is the name of your new virtual environment. Therefore, you can give any name for the virtual environment.

I have created a new directory named venv that contains the python standard library, the copy of the Python binary, the package manager of Pip, and all other supporting files.

Activate the virtual environment ‘venv’ by executing the below mentioned activate script:
```
$ source venv/bin/activate
```
Once the environment activated, you will see that at the beginning of the system

$PATH variable the bin directory of the virtual environment will be added. You will notice that the shell’s prompt name is changed now. The name of the currently using virtual environment will show on the shell’s prompt. Here, the name of the virtual environment in which we are currently working is ‘venv’.
### Update PIP
To install the TensorFlow, it’s required to first install the pip version 20 or more latest. The following command you can use to upgrade pip from previous to the latest version: (venv)
```
$ pip install --upgrade pip
```
### Installation of TensorFlow on Ubuntu 21.04
Once the virtual environment is activated, it’s time to start the installation of TensorFlow on your system. Type the following command to install the TensorFlow packages:
```
(venv) $ pip install --upgrade tensorflow
```
### Verify installation
To verify the installation of TensorFlow, execute the following command that will display the installed version of TensorFlow on the terminal:
```
$ python -c 'import tensorflow as tf; print(tf. version )'
```
### Update Tensorflow and Keras Using Pip
```
$ pip install -U tensorflow
```
Install matplotlib
```
$  pip install matplotlib
```
Install Scipy
```
$ pip install Scipy
```
