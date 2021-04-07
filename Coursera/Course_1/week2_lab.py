import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# Load the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()#(path='/Users/virajdatt.kohir/Downloads/train-labels-idx1-ubyte.gz')
#print(x_train.shape)

#print(np.unique(y_train))

#plt.imshow(x_train[0])

# Normalize the image

x_train = x_train / 255.0
x_test = x_test / 255.0

model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
])

## Custom CallBack

class CustomCallBack(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        print("Reached the end of epoch",  list(logs.keys()))

    def on_train_end(self, logs=None):
        print("Reached end of training", list(logs.keys()))


model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=2, callbacks=CustomCallBack())

model.evaluate(x_test, y_test)