import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
tf.config.set_visible_devices([], 'GPU')

AUTOTUNE = tf.data.AUTOTUNE

(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

IMG_SIZE = 150



resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])





resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])


def prepare(ds, shuffle=False, augment=False):
  # Resize and rescale all datasets.
  ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
              num_parallel_calls=AUTOTUNE)

  if shuffle:
    ds = ds.shuffle(1000)

  # Batch all datasets.
  ds = ds.batch(batch_size)

  # Use data augmentation only on the training set.
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=AUTOTUNE)


train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)


def format_image(image, label):
    image = tf.image.resize(image, (150, 150))/255.0
    return image, label


def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.map(format_image)
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(32)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

BATCH_SIZE = 32

train_ds = train_ds.map(format_image).shuffle(1000).batch(BATCH_SIZE).prefetch(1)

val_ds = val_ds.map(format_image).batch(BATCH_SIZE).prefetch(1)

test_ds = test_ds.map(format_image).batch(BATCH_SIZE).prefetch(1)

# train_ds = configure_for_performance(train_ds)
# val_ds = configure_for_performance(val_ds)
# test_ds = configure_for_performance(test_ds)


data_aug = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
                             tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                             tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
                             ])

val_aug = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
                            ])

aug_train_data = train_ds.map(lambda x, y: (data_aug(x, training=True), y))
aug_valid_data = val_ds.map(lambda x, y: (val_aug(x, training=True), y))
aug_test_ds = test_ds.map(lambda x, y: (val_aug(x, training=True), y))

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



model.fit(train_ds, #train_ds,
          epochs=20,
          #steps_per_epoch=100,
          validation_data=val_ds,
          #validation_steps=50,
          callbacks=[TB, MC, ES]
          )

for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0])
        plt.axis("off")