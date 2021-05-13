import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Fashion MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)


def make_train_set(bs=32):
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(100).batch(bs)
    dataset = dataset.prefetch(1)
    return dataset

def make_test_set():
    test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_set = test_set.batch(32)
    test_set = test_set.prefetch(1)
    return test_set


model_b32 = keras.Sequential([
    keras.layers.Convolution2D(32, (2,2), activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Convolution2D(64, (2,2), activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

dataset = make_train_set()
trainset = dataset.take(50000)
validset = tf.data.Dataset.from_tensor_slices((x_train[:50000], y_train[:50000])).batch(32).prefetch(1)


model_b32.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics='accuracy')

history_m32 = model_b32.fit(trainset,
                            validation_data=validset,
              #validation_data=(x_train[:50000], y_train[:50000]),
              epochs=2)

model_b32.evaluate(x_test, y_test)

def plot_acc(history):
    import matplotlib.pyplot as plt
    history_dict = history.history
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    acc = history_dict['accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc', color='green')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc', color='red')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_acc(history_m32)