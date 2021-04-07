'''
In the videos you looked at how you would improve Fashion MNIST using Convolutions. For your exercise see if you can improve MNIST to 99.8% accuracy or more using only a single convolutional layer and a single MaxPooling 2D. You should stop training once the accuracy goes above this amount. It should happen in less than 20 epochs, so it's ok to hard code the number of epochs for training, but your training must end once it hits the above metric. If it doesn't, then you'll need to redesign your layers.
When 99.8% accuracy has been hit, you should print out the string "Reached 99.8% accuracy so cancelling training!"
'''

from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()#(path='/Users/virajdatt.kohir/Downloads/train-labels-idx1-ubyte.gz')
print(x_train.shape)

# Reshape the training examples
x_train = x_train.reshape(60000, 28, 28, 1) # tensor, height, width, channles
x_test = x_test.reshape(10000, 28, 28, 1)
print("New Shape", type(x_train),x_train.shape)

# Normalize the examples

x_train = x_train / 255.0
x_test = x_test / 255.0

# Create your custom callback
class AccStop(keras.callbacks.Callback):
    '''
    Custom class to stop training once the Accuracy reaches 99.8% on training set
    '''

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= 0.998:
            print("The required accuracy reached, Stopping training")
            self.model.stop_training = True

# Build your model
model = keras.Sequential([
            keras.layers.Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'),
            keras.layers.MaxPool2D(2,2),
            # keras.layers.Conv2D(64, (2,2), activation='relu'),
            # keras.layers.MaxPool2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')

            ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],)

model.summary()

model.fit(x_train, y_train, callbacks=AccStop(), epochs=20, batch_size=128)

x_train.shape