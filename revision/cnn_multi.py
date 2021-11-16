import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
train_data = pd.read_csv('/Users/virajdatt/Desktop/TFCertPrep/data/sign-mnist/sign_mnist_train/sign_mnist_train.csv')
test_data = pd.read_csv('/Users/virajdatt/Desktop/TFCertPrep/data/sign-mnist/sign_mnist_test/sign_mnist_test.csv')

print(train_data.head(3))

train_labels = train_data['label'].values
test_labels = test_data['label'].values
# Preprocess the data
train_data.drop('label', axis=1, inplace=True)
test_data.drop('label', axis=1, inplace=True)

X_train = train_data.values.reshape(train_data.shape[0], 28, 28, 1)
X_test = test_data.values.reshape(test_data.shape[0], 28, 28, 1)

assert X_train[0].shape == (28, 28, 1)

X_train = X_train.astype('float64')
X_train = X_train/255.0

X_test = X_test.astype('float64')
X_test = X_test/255.0



print(len(np.unique(train_labels)))

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(64,  (2,2), activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(26, activation='softmax')
])
one_hot_encoded_train_labels = tf.one_hot(train_labels, depth=26)
one_hot_encoded_test_labels = tf.one_hot(test_labels, depth=26)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              #loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_datagen.flow(X_train, one_hot_encoded_train_labels, batch_size=32)
test_datagen.flow(X_test, one_hot_encoded_test_labels, batch_size=32)


model.fit(X_train, one_hot_encoded_train_labels, epochs=2)

model.evaluate(X_test, one_hot_encoded_test_labels)