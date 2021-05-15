import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
train = pd.read_csv('Data/sign_mnist/train.csv')
valid = pd.read_csv('Data/sign_mnist/test.csv')

train.head()
train_labels = train['label'].values
train = train.drop(['label'], axis=1)

valid_labels = valid['label'].values
valid = valid.drop(['label'], axis=1)


train_arr = train.values
valid_arr = valid.values

print(train_arr.shape)

train_arr = train_arr.reshape(train_arr.shape[0], *(28,28,1))
valid_arr = valid_arr.reshape(valid_arr.shape[0], *(28,28,1))

print(train_arr.shape)

plt.imshow(train_arr[0])


trainds = tf.data.Dataset.from_tensor_slices((train_arr, train_labels))
trainds = trainds.batch(32)


valids = tf.data.Dataset.from_tensor_slices((valid_arr, valid_labels))
valids = valids.batch(32)

for x,y in valids.take(1):
    print(x.shape, y)

data_aug = keras.Sequential([keras.layers.experimental.preprocessing.Rescaling(1. / 255),
                             keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                             keras.layers.experimental.preprocessing.RandomZoom(0.2),
                             #keras.layers.experimental.preprocessing.Rando
                             #keras.layers.experimental.preprocessing.RandomRotation(20)

                             ])

train_aug = trainds.map(lambda x,y: (data_aug(x), y))

for images, _ in trainds.take(1):
    for i in range(9):
        augmented_images = data_aug(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0])
        plt.axis("off")

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(trainds,
                   epochs=30,
                   validation_data=valids
                   )
