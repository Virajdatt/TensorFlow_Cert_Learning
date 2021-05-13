'''
The idea here is to experiment and record values for:-

1. SCCE vs CCE
2. TFDS vs Non-TFDS
3. TFDS_SCCE vs TFDS_CCE
'''

import tensorflow as tf
import numpy as np
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

# Data Prep
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

ohe_y_train = tf.one_hot(y_train, depth=10)
ohe_y_test = tf.one_hot(y_test, depth=10)

model_scce = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model_cce = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model_scce.compile(loss='sparse_categorical_crossentropy',
                   optimizer='adam',
                   metrics='accuracy'
                   )
model_scce.fit(x_train,
               y_train,
               epochs=10,
               verbose=0)

model_cce.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics='accuracy'
                  )

model_cce.fit(x_train,
              ohe_y_train,
              epochs=10,
              verbose=0)

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.batch(32)  # Don't wanna shuffle,
dataset = dataset.prefetch(2)

ohe_dataset = tf.data.Dataset.from_tensor_slices((x_train,ohe_y_train))
ohe_dataset = ohe_dataset.batch(32)  # Don't wanna shuffle,
ohe_dataset = ohe_dataset.prefetch(2)

model_tfds_cce = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model_tfds_cce.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics='accuracy'
                  )


model_tfds_cce.fit(ohe_dataset,
               epochs=10,
               verbose=0)



model_tfds_scce = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model_tfds_scce.compile(loss='sparse_categorical_crossentropy',
                   optimizer='adam',
                   metrics='accuracy'
                   )
model_tfds_scce.fit(dataset,
               epochs=10,
               verbose=0)

print('SCCE, ',model_scce.evaluate(x_test, y_test, verbose=0))
print('CCE, ',model_cce.evaluate(x_test, ohe_y_test, verbose=0))
print('TFDS SCCE,', model_tfds_scce.evaluate(x_test, y_test, verbose=0))
print('TFDS CCE,',model_tfds_cce.evaluate(x_test, ohe_y_test, verbose=0))

