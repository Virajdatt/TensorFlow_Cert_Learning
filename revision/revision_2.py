import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

## WEEK-1

### DNN for image classification on the MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print(f'The max pixel value in training data {np.max(x_train)},The min pixel value in training data {np.min(x_train)}')

x_train = x_train / 255.0
x_test = x_test/ 255.0

assert np.max(x_train) == 1
assert np.min(x_train) == 0

print(f'The max pixel value in training data {np.max(x_train)},The min pixel value in training data {np.min(x_train)}')

## We have preprocessed the data and made sure that the pixel values are between 0-1
## This step is taken in order to make sure that there is not a lot of variation in the input data
## which leads to inaccurate learnings in the DNN


### Model Building

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

class StopEpoch(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if (logs.get('accuracy') > 0.9):
            print(f'The required accuracy has reached, stopping training !!')
            self.model.stop_training = True




model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics= 'accuracy',
              )

history = model.fit(x_train, y_train,
                    epochs=40,
                    callbacks=StopEpoch())


plt.plot(np.diff(history.history['accuracy']))
plt.title('accuracy increment')
plt.xlabel('epochs')

def plot_acc(history):
    #import matplotlib.pyplot as plt
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




