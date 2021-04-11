"""
This is a self directed project to:-

1. Create image data using tf.dataset.
2. Use resnet50 CNN arch.
3. Use image augmentation.
4. Run a couple of experiments with Tboard for VIZ.
"""
from tensorflow import keras
import tensorflow as tf

"""
Expirement 1:- 

Cut down the resnet50 model at conv4_block4_out
"""
# creating tf.dataset for cats-v-dogs

train_ds = keras.preprocessing.image_dataset_from_directory('./Data/cats_and_dogs_filtered/train',
                                                            # subset='training',
                                                            seed=42,
                                                            image_size=(150, 150),
                                                            batch_size=20
                                                            )
valid_ds = keras.preprocessing.image_dataset_from_directory('./Data/cats_and_dogs_filtered/validation',
                                                            # subset='validation',
                                                            seed=42,
                                                            image_size=(150, 150),
                                                            batch_size=20
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

res50_model = keras.applications.ResNet50(input_shape=(150, 150, 3),
                                          include_top=False,
                                          weights='imagenet')
# https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

# res50_model.summary()

for layer in res50_model.layers:
    layer.trainable = False
last_layer = res50_model.get_layer("conv4_block4_out")
last_op = last_layer.output

x = keras.layers.Flatten()(last_op)
x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.Dropout(0.3)(x)
output = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(res50_model.input, output)

# model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )
tensorboard = keras.callbacks.TensorBoard('./files/run_resnet50_2nd')

model.fit(aug_train_data,
          epochs=3,
          steps_per_epoch=100,
          validation_data=aug_valid_data,
          validation_steps=50,
          callbacks=tensorboard
          )
