import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()#(path='/Users/virajdatt.kohir/Downloads/train-labels-idx1-ubyte.gz')
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
#print(x_train.shape)

#print(np.unique(y_train))

#plt.imshow(x_train[0])

# Normalize the image

x_train = x_train / 255.0
x_test = x_test / 255.0

model = keras.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28,28,1)),
        keras.layers.MaxPool2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPool2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
])





model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# TensorBoard

files_dir = os.path.join(os.curdir, 'files')

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(files_dir, run_id)
run_logdir = get_run_logdir()

tensorboard = keras.callbacks.TensorBoard(run_logdir)
model.fit(x_train, y_train, epochs=5, callbacks=tensorboard, validation_data=(x_test, y_test))


