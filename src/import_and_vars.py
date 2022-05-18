#import dei vari moduli utilizzati per l'implementazione del progetto
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Conv2D, MaxPool2D, SeparableConv2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

#size of input
single_size = 300

#number of channel for each image (grayscale = 1 canale, rgb = 3 canali)
color_mode = "rgb"
channel = 3 if color_mode == "rgb" else 1
target_size = (single_size, single_size)
input_size = (single_size, single_size, channel)

#size of the batch, number of epochs, number of classes and value of learning reate
batch_size = 128
num_epochs = 80
num_classes = 3
learning_rate = .0001

#path where saving the model
path_name = 'C:/Users/___/Desktop/dataset/'
save_path_name = 'C:/Users/___/Desktop/final_net/'
save_history_path_name = save_path_name + "history.csv"

#datagenerator for training set in which i define the value of data augmentation; furthermore i normalize the images with rescale
datagenerator_train = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=30,
    zoom_range=0.2,
    brightness_range = (0.7,1.3),
    horizontal_flip=True,
    fill_mode='nearest',
)

#datagenerator for test and validation, only normalization
datagenerator_test = ImageDataGenerator(rescale=1. / 255)

#creazione dei generatori per training, validation e test set
train_generator = datagenerator_train.flow_from_directory(
    directory = path_name + "/rps_train",
    target_size = target_size,
    color_mode = color_mode,
    batch_size = batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_generator = datagenerator_test.flow_from_directory(
    directory = path_name + "/rps_test",
    target_size = target_size,
    color_mode = color_mode,
    batch_size = batch_size,
    class_mode="categorical",
    shuffle=False,
)

val_generator = datagenerator_test.flow_from_directory(
    directory = path_name + "/rps_val",
    target_size = target_size,
    color_mode = color_mode,
    batch_size = batch_size,
    class_mode="categorical",
    shuffle=False,
)

#calculate train/validation/test steps depending on the size of the dataset
train_steps = train_generator.n // train_generator.batch_size
test_steps = test_generator.n // test_generator.batch_size
val_steps = val_generator.n // val_generator.batch_size