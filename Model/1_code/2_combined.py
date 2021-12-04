# coding: utf-8

# Introduction

"""
The Crowdsim, uses the baseline model predicts a binary label (malignant or not) from a skin lesion image.
The model is built on a convolutional base and extended further by adding specific layers.
As an encoder, we used the VGG16, Inception v3, and ResNet50 convolutional base. For this base,
containing a series of poolinig and convolution layers, we applied fixed pre-trained ImageNet weights.
We have trained the baseline n two ways: a) freeze the convolutional base
and train the rest of the layers and b) train all layers including the convolutional base.
"""

NAME = '2_combined'
FILEPATH = ''
PROJECT = 'CROWDSIM'
PYTHON_VERSION = '3.8.2'
KERAS_VERSION = '2.4.2'
TENSOR_FLOW_GPU = '2.2.0'

# Preamble
## Imports
from keras.initializers import glorot_uniform
import keras.backend.tensorflow_backend
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
from get_data import get_crowdsim_data

# Extra import for matplotlib
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

## Set working directory
workdir = re.sub("(?<={})[\w\W]*".format(PROJECT), "", os.getcwd())
os.chdir(workdir)

## Set up pipeline folder if missing
pipeline = os.path.join('empirical', '2_pipeline', NAME)
if not os.path.exists(pipeline):
    os.makedirs(pipeline)
    for folder in ['out', 'store', 'tmp']:
        os.makedirs(os.path.join(pipeline, folder))
        for network in ['vgg16', 'inception', 'resnet']:
            os.makedirs(os.path.join(pipeline, folder, network))



#Reading the model from JSON file
def load_model(seed): 
    global FILEPATH
    global result_path
    NAME = '0_baseline'

    pipeline = os.path.join('empirical', '2_pipeline', NAME)
    name = str(seed)+'base'

    if NETWORK_SELECTED == NETWORK_TYPE.VGG16:
        FILEPATH = os.path.join(pipeline, 'out', 'vgg16', name+'.json')
        result_path = os.path.join(pipeline, 'out', 'vgg16', str(seed)+'acc_and_loss.csv')
    else:
        if NETWORK_SELECTED == NETWORK_TYPE.INCEPTION:
            FILEPATH = os.path.join(pipeline, 'out', 'inception', name+'.json')
            result_path = os.path.join(pipeline, 'out', 'inception', str(seed)+'acc_and_loss.csv')
        else:
            FILEPATH = os.path.join(pipeline, 'out', 'resnet', name+'.json')
            result_path = os.path.join(pipeline, 'out', 'resnet', str(seed)+'acc_and_loss.csv')

    with open(FILEPATH) as json_file:
        json_savedModel= json_file.read()

    return json_savedModel


def read_data(seed):
    global test_id, test_label_c, class_weights, train, validation
    global train_id, train_label_c, valid_id, valid_label_c, test_id, test_label_c, class_weights

    # ground_truth_file = os.path.join('0_data', 'forVS', 'PH2_Ground_Truths.csv')
    ground_truth_file = r"C:/Users/COBOD/OneDrive/CBS doku/ITU/3semester/Project/GitHubs/Project/crowdsim/Extra/Data/ORACLE_2021_GroundTruth.csv"
    train_id, train_label_c, valid_id, valid_label_c, test_id, test_label_c, class_weights = get_crowdsim_data(ground_truth_file, seed, VERBOSE)

    # data_path = os.path.join('0_data', 'forVS')
    data_path = r"C:/Users/COBOD/OneDrive/CBS doku/ITU/3semester/Project/GitHubs/Project/crowdsim/Extra/Data/unlabeled_images/converted"
    train = generate_data_1(directory=data_path, augmentation=True, batchsize=BATCH_SIZE, file_list=train_id,
                            label_1=train_label_c)
    validation = generate_data_1(directory=data_path, augmentation=True, batchsize=BATCH_SIZE, file_list=valid_id,
                                 label_1=valid_label_c)

def fit_model(model):
    global history
    history = model.fit(
        train,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_steps=VALIDATION_STEPS,
        validation_data=validation,
        class_weight=class_weights)

def predict_model(model):
    ## Please note that augmentation is set to "True" like in Ralf's. and we use the VGG16 conv.
    test = generate_data_1(directory=r"C:/Users/COBOD/OneDrive/CBS doku/ITU/3semester/Project/GitHubs/Project/crowdsim/Extra/Data/unlabeled_images/converted", augmentation=True,
                           batchsize=BATCH_SIZE, file_list=test_id, label_1=test_label_c)
    predictions = model.predict_generator(test, steps=PREDICTION_STEPS)
    y_true = test_label_c
    delta_size = predictions.size - y_true.count()
    scores = np.resize(predictions, predictions.size - delta_size)

    filename = get_output_filename(str(seed)+'predictions.csv')
    df = pd.DataFrame({'id': test_id, 'prediction': scores, 'true_label': y_true})
    with open(filename, mode='w') as f:
        df.to_csv(f, index=False)

    auc = roc_auc_score(y_true, scores)
    return auc


def get_output_filename(name):
    if NETWORK_SELECTED == NETWORK_TYPE.VGG16:
        filename = os.path.join(pipeline, 'out', 'vgg16', name)
    else:
        if NETWORK_SELECTED == NETWORK_TYPE.INCEPTION:
            filename = os.path.join(pipeline, 'out', 'inception', name)
        else:
            filename = os.path.join(pipeline, 'out', 'resnet', name)
    return filename


def save_model(model, seed):
    model_json = model.to_json()

    filename = get_output_filename(str(seed)+'base')
    with open(filename + '.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(filename + '.h5')


## Start the model run
if VERBOSE:
    print_constants()

df_auc = pd.DataFrame(columns=['seed', 'auc'])
for seed in seeds:
    read_data(seed)

    #load the model architecture 
    json_savedModel = load_model(seed)
    model_comb = tf.keras.models.model_from_json(json_savedModel)
    print("Model loaded from JSON")

    # Load weights:
    weight_path = FILEPATH[:-4]+'h5'
    model_comb.load_weights((weight_path))
    print("Weight loaded")

    #Compiling the model
    model_comb.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])

    model_comb.summary()
    print("Model compiled")

    fit_model(model_comb)

    if SAVE_MODEL_WEIGHTS:
        save_model(model_comb, seed)
        print("Model saved")

    report_acc_and_loss(history, get_output_filename(str(seed)+'acc_and_loss.csv'))

    score = predict_model(model_comb)
    df_auc = df_auc.append({'seed': seed, 'auc': score}, ignore_index=True)

report_auc(df_auc, get_output_filename('aucs.csv'))

