from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf
import numpy as np
housing_data = datasets.fetch_california_housing()

x_train, x_test, y_train,  y_test = train_test_split(housing_data.data, housing_data.target)

xt, xv, yt, yv = train_test_split(x_train, y_train)

scaler = StandardScaler()
xt = scaler.fit_transform(xt)
x_test = scaler.transform(x_test)
xv = scaler.transform(xv)


xt[1:].shape
xt.shape
# Custom Loss Functions:-

# def create_huber(threshold=1.0):
#
#     def huber_fn(y_true, y_pred):
#
#         error = y_true - y_pred
#         is_small_error = tf.abs(error) < threshold
#         squared_loss = tf.square(error) / 2
#         linear_loss = threshold * tf.abs(error) - threshold**2 / 2
#         return tf.where(is_small_error, squared_loss, linear_loss)
#     return huber_fn



class Huber_Loss(keras.losses.Loss):

    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold ** 2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        base_config = super.get_config()
        return {**base_config, 'threshold': self.threshold}

model = keras.Sequential([keras.layers.Dense(30, activation='relu', input_shape=(11610,8)),
                          keras.layers.Dense(20, activation='relu'),
                          keras.layers.Dense(1)])
model.compile(loss=Huber_Loss(),
              optimizer='adam',
              metrics=['mse'])

model.summary()
model.fit(xt, yt, epochs=2, validation_data=(xv, yv))


model.evaluate(xt, yt)

from tensorflow.keras import  backend as K
K.clear_session()