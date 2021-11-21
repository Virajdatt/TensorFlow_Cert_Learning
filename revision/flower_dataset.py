import tensorflow as tf

tf.keras.backend.clear_session()


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1. / 255)

train_images = train_datagen.flow_from_directory('/Users/virajdatt/Desktop/TFCertPrep/data/flower_photos',
                   batch_size=20,
                   target_size=(150, 150),
                   )

#train_images.class_names

cat_dog_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Conv2D(128, (2,2), activation='relu'),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(5, activation='softmax')
])

cat_dog_model.compile(loss='categorical_crossentropy',
                      metrics=['accuracy'],
                      optimizer='adam'
                      )

#tf.keras.utils.plot_model(cat_dog_model)
class StopTrain(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('accuracy')>0.9):
            print('Target accuracy reached, stopping training')
            self.model.stop_training = True


TB = tf.keras.callbacks.TensorBoard('/Users/virajdatt/Desktop/TFCertPrep/files/flowers_vanila')



history = cat_dog_model.fit(
      train_images,
      steps_per_epoch=100,  # 2000 images = batch_size * steps
      epochs = 20,
      #validation_data=validation_generator,
     # validation_steps=50,  # 1000 images = batch_size * steps
      verbose=2,
      callbacks=[StopTrain(), TB])


