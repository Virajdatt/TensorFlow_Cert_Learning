import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Part 1 just tfds

# Load the data

(train, test), metadata = tfds.load('mnist',
                                    as_supervised=True,
                                    with_info=True,
                                    split=['train', 'test'])


# Preprocess
# a. Rescale
rescale = keras.layers.experimental.preprocessing.Rescaling(1./255)

trainds = train.map(lambda x,y: (rescale(x, training=True), y))
testds = test.map(lambda  x,y:( rescale(x, training=True), y))
#
# for x,y in trainds.skip(2).take(1):
#
#     plt.title(str(y.numpy()))
#     plt.imshow(x)


trainds = trainds.shuffle(1000).batch(32).prefetch(1)
valids = testds.batch(32).prefetch(1)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(2),
    #keras.layers.Dropout(.2),

    keras.layers.Conv2D(64, (3, 3), activation='relu',),
    keras.layers.MaxPool2D(2),
    #keras.layers.Dropout(.2),


    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')

])

MC = keras.callbacks.ModelCheckpoint(
    './Models/mnist_tfds/mnist_h5.h5',
    monitor='val_loss',
    save_best_only='True',
    verbose=1
)

ES = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights='True'
)

LR = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10 ** (epoch/2), verbose=1)
TB = keras.callbacks.TensorBoard('./Models/tb_logs/mnist_tfds')

ROP = keras.callbacks.ReduceLROnPlateau(
    patience=2,
    verbose=1,
    factor=0.01,
    min_delta=0.12,
    min_lr=0.01
)

model.compile(loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              optimizer=keras.optimizers.Adam(learning_rate=1e-5))

#model.summary()

model.fit(
    trainds,
    validation_data=valids,
    callbacks=[ES, MC, ROP, ],
    epochs=4
)


# model.evaluate(valids)
#
# saved_model = keras.models.load_model('./Models/mnist_tfds')
#
# saved_model.save('./Models/mnist_tfds.h5')
#
#
# loading_model = keras.models.load_model('./Models/mnist_tfds.h5')
#
# loading_model.evaluate(valids)

# print(1e-5 * 10 **(2/2))


# Part 2 arrays

train_array = []
train_label = []

test_array = []
test_label = []

for x,y in train:
    train_array.append(x.numpy())
    train_label.append(y.numpy())

for x,y in test:
    test_array.append(x.numpy())
    test_label.append(y.numpy())

train_array = np.array(train_array)
train_label = np.array(train_label)
test_array = np.array(test_array)
test_label = np.array(test_label)

print(np.max(train_array), np.min(train_array))
train_array2 = train_array / 255.0
print(np.max(train_array2), np.min(train_array2))

test_array2 = test_array / 255.0


model2 = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(2),
    #keras.layers.Dropout(.2),

    keras.layers.Conv2D(64, (3, 3), activation='relu',),
    keras.layers.MaxPool2D(2),
    #keras.layers.Dropout(.2),


    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')

])

MC = keras.callbacks.ModelCheckpoint(
    './Models/mnist_tfds/mnist_array_h5.h5',
    monitor='val_loss',
    save_best_only='True',
    verbose=1
)

ES = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    verbose=1,
    restore_best_weights='True'
)

LR = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10 ** (epoch/2), verbose=1)
TB = keras.callbacks.TensorBoard('./Models/tb_logs/mnist_array')

model2.compile(loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              optimizer=keras.optimizers.Adam(learning_rate=1e-5))

#model.summary()

model2.fit(
    train_array2, train_label,
    validation_data=(test_array2, test_label),
    callbacks=[ES, MC, LR, TB],
    epochs=4
)

model.evaluate(valids)
model2.evaluate(test_array2, test_label)