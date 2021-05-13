import tensorflow as tf
from tensorflow import keras
import numpy as np
# MNIST

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess:-

x_train = x_train / 255.0
x_test = x_test / 255.0

ohe_ytrain = tf.one_hot(y_train, depth=10)
ohe_y_test = tf.one_hot(y_test, depth=10)


class custom_cb(keras.callbacks.Callback):

    def on_train_batch_end(self, batch, logs=None):
        print(logs.keys())

    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('accuracy') > 0.999):
            print('Accuracy crossed 80 %')
            self.model.stop_training = True


model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
    #loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

model.fit(x_train,
          ohe_ytrain,
          callbacks=custom_cb(),
          epochs=10)

model.evaluate(x_test, ohe_y_test)

