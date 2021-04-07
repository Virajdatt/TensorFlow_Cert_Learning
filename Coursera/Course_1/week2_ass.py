from tensorflow import keras
# import matplotlib.pyplot as plt

# MNIST

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#plt.imshow(x_train[0])

x_train = x_train / 255.0
x_test = x_test / 255.0

class AccStop(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        if (logs.get('accuracy') > 0.95):
            print("90 % training accuracy reached")
            self.model.stop_training = True

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(512, activation='relu'),
                          keras.layers.Dense(10, activation='softmax')
                          ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=AccStop())