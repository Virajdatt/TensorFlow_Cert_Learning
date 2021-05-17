import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

(trainds, valids, testset), metadata = tfds.load('tf_flowers',
                                       split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                       with_info=True,
                                       as_supervised=True)
print(
tf.data.experimental.cardinality(trainds) + tf.data.experimental.cardinality(valids) + \
tf.data.experimental.cardinality(testset))


label_string = metadata.features['label'].int2str

count=1
for x, y in trainds:
    if count == 6:
        break
    plt.figure(count)
    plt.imshow(x)
    plt.title(label_string(y))

    count = count+1

IMG_SIZE = 180
AUTOTUNE = tf.data.AUTOTUNE

rescale_resize = keras.Sequential([
    keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
    keras.layers.experimental.preprocessing.Rescaling(1./255)
])


trainds = trainds.map(lambda x, y: (rescale_resize(x, training=True), y))
                      #num_parallel_calls=AUTOTUNE)
valids = valids.map(lambda x, y: (rescale_resize(x, training=True), y))
testset = testset.map(lambda x, y: (rescale_resize(x, training=True), y))

count=1
for x, y in trainds:
    if count == 6:
        break
    plt.figure(count)
    plt.imshow(x)
    plt.title(label_string(y))

    count = count+1

data_aug = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    keras.layers.experimental.preprocessing.RandomZoom(0.1),
    keras. layers.experimental.preprocessing.RandomRotation(0.2),

])

def create_ds(ds,shuffle=False, bs=32):
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(bs)
    ds = ds.prefetch(AUTOTUNE)
    return ds


trainds = create_ds(trainds, shuffle=True)
valids = create_ds(valids)
testset = create_ds(testset)

for images, _ in trainds.take(1):
    for i in range(9):
        augmented_images = data_aug(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0])
        plt.axis("off")

train_aug = trainds.map(lambda x,y: (data_aug(x,training=True),y))

for x,y in train_aug.take(1):
    print(x.shape, y.shape)

model = keras.Sequential([
    keras.layers.Conv2D(16, 2, padding='same', activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(32, 2, padding='same', activation='relu',),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, 2, padding='same', activation='relu', ),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(5, activation='softmax'),

])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy']
              )

history = model.fit(train_aug,
          validation_data=valids,
          epochs=10)
