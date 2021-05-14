import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from tensorflow.keras.applications.inception_v3 import InceptionV3

tpath = 'Data/horses-humans/train'
testpath = 'Data/horses-humans/test'

tpath_horses = tpath+'/horses/'
tpath_humans = tpath+'/humans/'


files_train_horses = os.listdir(tpath_horses)
files_train_humans = os.listdir(tpath_humans)


img = load_img(tpath_humans+files_train_humans[0])
img = img_to_array(img)
print(img.shape)
plt.imshow(img.astype('uint8'))

seed = 69
bs = 32

trainds = keras.preprocessing.image_dataset_from_directory('Data/horses-humans/train',
                                                           image_size=(200, 200),
                                                           seed=seed,
                                                           shuffle=True,
                                                           batch_size=bs)


valids = keras.preprocessing.image_dataset_from_directory('Data/horses-humans/test',
                                                           image_size=(200, 200),
                                                           seed=seed,
                                                           batch_size=bs)


data_aug = keras.Sequential([keras.layers.experimental.preprocessing.Rescaling(1. / 255),
                             keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                             keras.layers.experimental.preprocessing.RandomZoom(0.2),
                             #keras.layers.experimental.preprocessing.RandomRotation(20)

                             ])

val_aug = keras.Sequential([keras.layers.experimental.preprocessing.Rescaling(1. / 255),
                            ])

train_aug = trainds.map(lambda x,y: (data_aug(x), y))

valid_aug = valids.map(lambda x,y: (val_aug(x), y))

card = tf.data.experimental.cardinality(valid_aug)
test_set = valid_aug.take(card//4)
valid_aug = valid_aug.skip(card//4)

for images, _ in trainds.take(1):
    for i in range(9):
        augmented_images = data_aug(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0])
        plt.axis("off")


inv_model = InceptionV3(include_top=False,
                        weights='imagenet',
                        input_shape=(200, 200, 3))

inv_model.summary()

for layer in inv_model.layers:
    layer.trainable = False

inv_model.summary()

last_layer = inv_model.get_layer('mixed5')
last_output = last_layer.output

x = keras.layers.Flatten()(last_output)
x = keras.layers.Dense(512, activation='relu')(x)
x = keras.layers.Dropout(0.25)(x)
output = keras.layers.Dense(1, activation='relu')(x)

model = keras.models.Model(inv_model.input, output)
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_aug,
                    validation_data=valid_aug,
                    epochs=5)