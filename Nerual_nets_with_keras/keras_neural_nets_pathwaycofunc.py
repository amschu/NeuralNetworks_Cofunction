"""
CREATED BY: Ally Schumacher CODE
ADOPTED FROM: Chapter 10:
https://github.com/ageron/handson-ml2/blob/master/10_neural_nets_with_keras.ipynb

DATE CREATED: 03/14/22
"""

import sys, os, argparse
import sklearn
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
import datatable as dt


#################
### LOAD DATA ###
#################

df = dt.fread("Nerual_nets_with_keras/3pathways_nn.csv", sep=',')
df = df.to_pandas()
df = df.set_index(df.columns[0], drop=True)


######################
### PRE-PROCESSING ###
######################

df['Y'] = pd.to_numeric(df['Y'], errors='raise')
X_train = df.drop(['Y'], axis=1)
y_train = df['Y']

X_test = test_df.drop(['Y'], axis=1)
y_test = test_df['Y']

print("x_train shape:", '\n', X_train.head())
print("y_train shape:", '\n', y_train.head())
print("X_test shape:", '\n', X_test.head())
print("y_test shape:", '\n', y_test.head())
#############################
### WHERE TO SAVE FIGURES ###
#############################

# PROJECT_ROOT_DIR = "."
# CHAPTER_ID = "ann"
# IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
# os.makedirs(IMAGES_PATH, exist_ok=True)
#
# def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
#     path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
#     print("Saving figure", fig_id)
#     if tight_layout:
#         plt.tight_layout()
#     plt.savefig(path, format=fig_extension, dpi=resolution)

##################
### PERCEPTRON ###
##################
