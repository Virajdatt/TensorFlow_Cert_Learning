import os
import zipfile
import random
import tensorflow as tf
from shutil import copyfile



#os.rmdir()

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[:testing_length]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)

CAT_SOURCE_DIR = "data/pet_data/PetImages/Cat/"
TRAINING_CATS_DIR = "data/pet_data/train/cats/"
TESTING_CATS_DIR = "data/pet_data/test/cats/"
DOG_SOURCE_DIR = "data/pet_data/PetImages/Dog/"
TRAINING_DOGS_DIR = "data/pet_data/train/dogs/"
TESTING_DOGS_DIR = "data/pet_data/test/dogs/"


split_size = .9
# split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
# split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'data/pet_data/train/',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        'data/pet_data/test/',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

cat_dog_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Conv2D(128, (2,2), activation='relu'),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
])

cat_dog_model.compile(loss='binary_crossentropy',
                      metrics=['accuracy'],
                      optimizer='adam'
                      )

#tf.keras.utils.plot_model(cat_dog_model)
class StopTrain(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('accuracy')>0.9):
            print('Target accuracy reached, stopping training')
            self.model.stop_training = True

history = cat_dog_model.fit(
      train_generator,
      steps_per_epoch=100,  # 2000 images = batch_size * steps
      epochs = 20,
      validation_data=validation_generator,
      validation_steps=50,  # 1000 images = batch_size * steps
      verbose=2,
      callbacks=StopTrain())


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

plot_acc(history)