import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from tensorflow.keras.applications.inception_v3 import InceptionV3

seed = 69
bs = 32

trainds = keras.preprocessing.image_dataset_from_directory('Data/rps/train',
                                                           image_size=(150, 150),
                                                           seed=seed,
                                                           shuffle=True,
                                                           batch_size=bs)


valids = keras.preprocessing.image_dataset_from_directory('Data/rps/test',
                                                           image_size=(150, 150),
                                                           seed=seed,
                                                           batch_size=bs)


data_aug = keras.Sequential([keras.layers.experimental.preprocessing.Rescaling(1. / 255),
                             keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                             keras.layers.experimental.preprocessing.RandomZoom(0.2),
                             #keras.layers.experimental.preprocessing.RandomRotation(40),
                             #keras.layers.experimental.preprocessing.RandomWidth(0.2),

                             #keras.layers.experimental.preprocessing.RandomRotation(20)

                             ])

val_aug = keras.Sequential([keras.layers.experimental.preprocessing.Rescaling(1. / 255),
                            ])

train_aug = trainds.map(lambda x,y: (data_aug(x), y))

valid_aug = valids.map(lambda x,y: (val_aug(x), y))


# for images, _ in train_aug.take(1):
#     for i in range(9):
#         augmented_images = images#data_aug(images)
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(augmented_images[0])
#         plt.axis("off")


# for x,y in trainds.take(1):
#     print(y.numpy())

sparse_train_aug = train_aug.map(lambda x,y: (x, tf.one_hot(y, depth=3)))
sparse_valid_aug = valid_aug.map(lambda x,y: (x, tf.one_hot(y, depth=3)))


# for x,y in sparse_train_aug.take(1):
#     print(y[0].numpy())
#     plt.title(y[0].numpy())
#     plt.imshow(x[0].numpy())

def cnn_maxpool(cnn_units=32,):
  return [keras.layers.Convolution2D(cnn_units, (3,3), activation='relu'),
          keras.layers.BatchNormalization(),
  keras.layers.MaxPool2D(),]

# model = keras.Sequential([
#     keras.layers.Convolution2D(64, (3,3), activation='relu', input_shape=(150, 150,)),
#     keras.layers.MaxPool2D(2),
#     *cnn_maxpool(64),
#     *cnn_maxpool(128),
#     *cnn_maxpool(128),
#     keras.layers.Flatten(),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(512, activation='relu'),
#     keras.layers.Dense(3, activation='softmax')
# ])
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

history = model.fit(sparse_train_aug,
          validation_data=sparse_valid_aug,
          epochs=1,
          verbose=1)

for x,y in sparse_valid_aug:
    print(x.shape)
    break