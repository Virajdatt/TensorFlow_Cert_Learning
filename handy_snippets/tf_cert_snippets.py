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

# Vizualizing intermediate activation

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

img = load_img('/tmp/h-or-s/sad/sad1-00.png', target_size=(150, 150))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in model.layers]

for layer_name, feature_map in zip(layer_names, successive_feature_maps):

  if len(feature_map.shape) == 4:
    print(layer_name)
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')



# Creating image dataset 1.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
# --------------------
# Flow validation images in batches of 20 using test_datagen generator
# --------------------
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))



# Creating TF DataSet for image data

train_ds = keras.preprocessing.image_dataset_from_directory('./Data/cats_and_dogs_filtered/train',
                                                            # subset='training',
                                                            seed=42,
                                                            image_size=(150, 150),
                                                            batch_size=20,
                                                            shuffle=
                                                            )
valid_ds = keras.preprocessing.image_dataset_from_directory('./Data/cats_and_dogs_filtered/validation',
                                                            # subset='validation',
                                                            seed=42,
                                                            image_size=(150, 150),
                                                            batch_size=20,
                                                            shuffle=

                                                            )

train_ds = train_ds.prefetch(buffer_size=32)
valid_ds = valid_ds.prefetch(buffer_size=32)

data_aug = keras.Sequential([keras.layers.experimental.preprocessing.Rescaling(1. / 255),
                             keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                             keras.layers.experimental.preprocessing.RandomZoom(0.2)

                             ])

val_aug = keras.Sequential([keras.layers.experimental.preprocessing.Rescaling(1. / 255),
                            ])

aug_train_data = train_ds.map(lambda x, y: (data_aug(x, training=True), y))
aug_valid_data = valid_ds.map(lambda x, y: (val_aug(x, training=True), y))


# Image Agumentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/tmp/horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/tmp/validation-horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Vizualize the augmentation
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


def cnn_maxpool(cnn_units=32,):
  return [keras.layers.Convolution2D(cnn_units, (2,2), activation='relu'),
  keras.layers.MaxPool2D(),
  keras.layers.BatchNormalization()]

*cnn_maxpool(32),