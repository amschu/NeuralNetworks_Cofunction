"""
ADOPTED FROM THE TUTORIAL:
"""

# Python ≥3.5 is required
import sys

# Scikit-Learn ≥0.20 is required
import sklearn

# TensorFlow ≥2.0 is required
import tensorflow as tf

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ann"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

# iris = load_iris()
# X = iris.data[:, (2, 3)]  # petal length, petal width
# y = (iris.target == 0).astype(np.int)
#
# per_clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
# per_clf.fit(X, y)
#
# y_pred = per_clf.predict([[2, 0.5]])
# print(y_pred)
#
# a = -per_clf.coef_[0][0] / per_clf.coef_[0][1]
# b = -per_clf.intercept_ / per_clf.coef_[0][1]
#
# axes = [0, 5, 0, 2]
#
# x0, x1 = np.meshgrid(
#     np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
#     np.linspace(axes[2], axes[3], 200).reshape(-1, 1), )
#
# X_new = np.c_[x0.ravel(), x1.ravel()]
# y_predict = per_clf.predict(X_new)
# zz = y_predict.reshape(x0.shape)
#
# plt.figure(figsize=(10, 4))
# plt.plot(X[y == 0, 0], X[y == 0, 1], "bs", label="Not Iris-Setosa")
# plt.plot(X[y == 1, 0], X[y == 1, 1], "yo", label="Iris-Setosa")
#
# plt.plot([axes[0], axes[1]], [a * axes[0] + b, a * axes[1] + b], "k-",
#          linewidth=3)
# from matplotlib.colors import ListedColormap
#
# custom_cmap = ListedColormap(['#9898ff', '#fafab0'])
#
# plt.contourf(x0, x1, zz, cmap=custom_cmap)
# plt.xlabel("Petal length", fontsize=14)
# plt.ylabel("Petal width", fontsize=14)
# plt.legend(loc="lower right", fontsize=14)
# plt.axis(axes)
#
# save_fig("perceptron_iris_plot")
# plt.show()

################################################################################
import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
# All sets created are numpy arrays
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# looking at the data in more detail
print("X_train_full", '\n', X_train_full.shape)
print("X_test", '\n', X_test.shape)

### Creating a validation set
# GRADIENT DESCENT will be used to train the model
# there must be a range of 0-1 and then you convert to floats after dividing
# them by 255.0

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# create class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
               "Shirt", "Sneaker", "Bag", "Ankle boot"]

###############################################
### CREATING THE MODEL USING SEQUENTIAL API ###
###############################################

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

import pydot

model.summary()
keras.utils.plot_model(model)