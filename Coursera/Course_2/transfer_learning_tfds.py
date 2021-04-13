# import tensorflow as tf
import os
from tensorflow import keras
import tensorflow_datasets as tfds

#Create a TF Dataset

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

tfds.core.benchmark(train_ds)
data_aug = keras.Sequential([keras.layers.experimental.preprocessing.Rescaling(1. / 255),
                             keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                             keras.layers.experimental.preprocessing.RandomZoom(0.2)

                             ])

val_aug = keras.Sequential([keras.layers.experimental.preprocessing.Rescaling(1. / 255),
                            ])

aug_train_data = train_ds.map(lambda x, y: (data_aug(x, training=True), y))
aug_valid_data = valid_ds.map(lambda x, y: (val_aug(x, training=True), y))

# 1. Instantiate the model you want to use

pre_trained_model = keras.applications.InceptionV3(input_shape=(150, 150,3),
                                                   include_top=False,
                                                   weights=None)

# Download the pretrained weights and load them into the keras model

# !wget --no-check-certificate \
#     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
#     -O ./Weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

pre_trained_model.load_weights('./Weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

#pre_trained_model.summary()
# pre_trained_model.get_weights()

# 2. Freeze layers of your model

for layer in pre_trained_model.layers:
    layer.trainable = False


# 3. Add your layers to the pretrained model

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

# Concatenate your new layers with the existing ones
out_put_classes = 1
x = keras.layers.Flatten()(last_output)
x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
if out_put_classes > 2:
    moutput = keras.layers.Dense(out_put_classes,activation='softmax' )(x)
else:
    moutput = keras.layers.Dense(out_put_classes, activation='sigmoid')(x)

model = keras.Model(pre_trained_model.input, moutput)#<KerasTensor: shape=(None, 150, 150, 3) dtype=float32 (created by layer 'input_1')>, x
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# Look at what is trainable
for layer in model.layers:
    if layer.trainable == True:
        print(layer)
# <tensorflow.python.keras.layers.core.Flatten object at 0x103b36040>
# <tensorflow.python.keras.layers.core.Dense object at 0x15aa483a0>
# <tensorflow.python.keras.layers.core.Dropout object at 0x110a52f70>

# 4 Prepare your data and fit the model

# wget --no-check-certificate \
#         https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
#        -O ./Data/cats_and_dogs_filtered.zip


files_dir = os.path.join(os.curdir, 'files')

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(files_dir, run_id)
run_logdir = get_run_logdir()

tensorboard = keras.callbacks.TensorBoard(run_logdir)


model.fit(aug_train_data,
          epochs=5,
          steps_per_epoch=100,
          validation_data=aug_valid_data,
          validation_steps=50,
          callbacks=tensorboard
          )