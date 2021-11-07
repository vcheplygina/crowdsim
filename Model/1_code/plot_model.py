## Plotting the model by using the saved json file
# Inspiration from: https://towardsdatascience.com/saving-and-loading-keras-model-42195b92f57a

## Imports
from keras.initializers import glorot_uniform
import os, re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

from constants import *
from report_results import report_acc_and_loss, report_auc
from sklearn.metrics import roc_auc_score
from generate_data import generate_data_1
from get_data import get_baseline_data

## Defined variables
NAME = '0_baseline'
SEED = 1972
FILEPATH = ''
PROJECT = 'CROWDSIM'
PYTHON_VERSION = '3.8.2'
KERAS_VERSION = '2.4.2'
TENSOR_FLOW_GPU = '2.2.0'

#Reading the model from JSON file
def load_model(seed): 
    global FILEPATH
    global result_path

    pipeline = os.path.join('empirical', '2_pipeline', NAME)
    name = str(seed)+'base'

    if NETWORK_SELECTED == NETWORK_TYPE.VGG16:
        FILEPATH = os.path.join(pipeline, 'out', 'vgg16', name+'.json')
        result_path = os.path.join(pipeline, 'out', 'vgg16', str(SEED)+'acc_and_loss.csv')
    else:
        if NETWORK_SELECTED == NETWORK_TYPE.INCEPTION:
            FILEPATH = os.path.join(pipeline, 'out', 'inception', name+'.json')
            result_path = os.path.join(pipeline, 'out', 'inception', str(SEED)+'acc_and_loss.csv')
        else:
            FILEPATH = os.path.join(pipeline, 'out', 'resnet', name+'.json')
            result_path = os.path.join(pipeline, 'out', 'resnet', str(SEED)+'acc_and_loss.csv')

    with open(FILEPATH) as json_file:
        json_savedModel= json_file.read()

    return json_savedModel


def read_data(seed):
    global test_id, test_label_c, class_weights, train, validation
    global train_id, train_label_c, valid_id, valid_label_c, test_id, test_label_c, class_weights

    # ground_truth_file = os.path.join('0_data', 'forVS', 'PH2_Ground_Truths.csv')
    ground_truth_file = r"C:/Users/COBOD/3D Objects/ISIC-2017_Training_Data/ISIC-2017_Training_Part3_GroundTruth.csv"
    train_id, train_label_c, valid_id, valid_label_c, test_id, test_label_c, class_weights = get_baseline_data(ground_truth_file, seed, VERBOSE)

    # data_path = os.path.join('0_data', 'forVS')
    data_path = r"C:/Users/COBOD/3D Objects/ISIC-2017_Training_Data"
    train = generate_data_1(directory=data_path, augmentation=False, batchsize=BATCH_SIZE, file_list=train_id,
                            label_1=train_label_c)
    validation = generate_data_1(directory=data_path, augmentation=False, batchsize=BATCH_SIZE, file_list=valid_id,
                                 label_1=valid_label_c)

#load the model architecture 
json_savedModel = load_model(SEED)
model_j = tf.keras.models.model_from_json(json_savedModel)
# model_j.summary()

# Load weights:
weight_path = FILEPATH[:-4]+'h5'
model_j.load_weights((weight_path))

#Compiling the model
model_j.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])

# Loading test set and fit model
read_data(SEED)

history= pd.read_csv(result_path)
# summarize history for accuracy
print(history)
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print("wait")






