import tensorflow as tf
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

tf.config.set_visible_devices([], 'GPU')

train_ds = tf.keras.preprocessing.image_dataset_from_directory('/Users/virajdatt/Desktop/TFCertPrep/data/flower_photos',
                                                             subset='training',
                                                            seed=42,
                                                            image_size=(150, 150),
                                                            batch_size=20,
                                                            validation_split=0.2
                                                            )
valid_ds = tf.keras.preprocessing.image_dataset_from_directory('/Users/virajdatt/Desktop/TFCertPrep/data/flower_photos',
                                                            subset='validation',
                                                            seed=42,
                                                            image_size=(150, 150),
                                                            batch_size=20,
                                                            validation_split=0.2
                                                            )
train_ds = train_ds.prefetch(buffer_size=32)
valid_ds = valid_ds.prefetch(buffer_size=32)

data_aug = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
                             tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                             tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
                             ])

val_aug = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
                            ])

aug_train_data = train_ds.map(lambda x, y: (data_aug(x, training=True), y))
aug_valid_data = valid_ds.map(lambda x, y: (val_aug(x, training=True), y))


model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(16, (2,2), activation='relu', input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Conv2D(32, (2,2), activation='relu'),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Conv2D(128, (2, 2), activation='relu'),
                tf.keras.layers.MaxPool2D(2, 2),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'],
                      optimizer='adam'
                      )

#tf.keras.utils.plot_model(model)
class StopTrain(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('accuracy')>0.9):
            print('Target accuracy reached, stopping training')
            self.model.stop_training = True

MC = tf.keras.callbacks.ModelCheckpoint(
    '/Users/virajdatt/Desktop/TFCertPrep/model/flowers.h5',
    monitor='val_loss',
    save_best_only='True',
    verbose=1
)

ES = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights='True'
)


TB = tf.keras.callbacks.TensorBoard('/Users/virajdatt/Desktop/TFCertPrep/files/flowers_tfds_dout')



model.fit(aug_train_data, #train_ds,
          epochs=20,
          #steps_per_epoch=100,
          validation_data=aug_valid_data,
          #validation_steps=50,
          callbacks=[TB, MC, ES]
          )

model.evaluate(aug_valid_data)

loaded_model = tf.keras.models.load_model('/Users/virajdatt/Desktop/TFCertPrep/model/flowers.h5')
loaded_model.evaluate(aug_valid_data)