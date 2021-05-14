from tensorflow import keras
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Read in the data
bs = 20

trainds = keras.preprocessing.image_dataset_from_directory('./Data/cats_and_dogs_filtered/train',
                                                           image_size=(150, 150),
                                                           seed=69,
                                                           batch_size=bs,
                                                           )

valids = keras.preprocessing.image_dataset_from_directory('./Data/cats_and_dogs_filtered/validation',
                                                           image_size=(150, 150),
                                                           seed=69,
                                                           batch_size=bs
                                                           )

aug_train = keras.Sequential([
    keras.layers.experimental.preprocessing.Rescaling(1./255),
    keras.layers.experimental.preprocessing.RandomZoom(0.2),
    keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    keras.layers.experimental.preprocessing.RandomZoom(0.1)
])

aug_val = keras.Sequential([
    keras.layers.experimental.preprocessing.Rescaling(1./255)
])

aug_trainds = trainds.map(lambda x,y:(aug_train(x, training=True), y))
aug_valids = valids.map(lambda x,y:(aug_val(x, training=True), y))


# Create a Test Set

print(trainds.class_names)
val_cardinality = tf.data.experimental.cardinality(valids)
testset = aug_valids.take(val_cardinality//5)
aug_valids = aug_valids.skip(val_cardinality//5)
print(tf.data.experimental.cardinality(aug_valids))



plt.figure(figsize=(10, 10))

for images, labels in aug_trainds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(int(labels[i]))
        #plt.axis("off")


for images, _ in trainds.take(1):
    for i in range(9):
        augmented_images = aug_train(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0])
        plt.axis("off")

for images, _ in valids.take(1):
    for i in range(9):
        augmented_images = aug_val(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0])
        plt.axis("off")


# Transfer Learning part

tl_model = InceptionV3(input_shape=(150, 150, 3),
                       weights='imagenet',
                       include_top=False,
                       )

tl_model.summary()

for layer in tl_model.layers:
    layer.trainable = False

tl_model.summary()

last_layer = tl_model.get_layer('mixed5')
print(last_layer.output)
last_output = last_layer.output

# Build our Model part after clipping of the original model

x = keras.layers.Flatten()(last_output)
x = keras.layers.Dense(512, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
output = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.models.Model(tl_model.input, output)
model.summary()


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(trainds,
          validation_data=valids,
          epochs=1)



