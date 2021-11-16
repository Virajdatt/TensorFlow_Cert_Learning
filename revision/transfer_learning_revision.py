import tensorflow as tf
import ssl
from tensorflow import keras

ssl._create_default_https_context = ssl._create_unverified_context


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

# inceptionv3 = tf.keras.applications.InceptionV3(input_shape=(150, 150, 3),
#                                            include_top=False,
#                                            weights='imagenet'
#                                            )
#
# inceptionv3.summary()
#
# for layers in inceptionv3.layers:
#     layers.trainable = False
#
# last_layer = inceptionv3.get_layer('mixed7')
#
# inceptionv3_last = last_layer.output
#
# new_model = tf.keras.layers.Flatten()(inceptionv3_last)
# new_model = tf.keras.layers.Dense(1024, activation='relu') (new_model)
# new_model = tf.keras.layers.Dropout(0.2) (new_model)
# new_model = tf.keras.layers.Dense(1, activation='sigmoid') (new_model)
#
# train_model = tf.keras.Model(inceptionv3.input, new_model)
#
# train_model.compile(optimizer='adam',
#                     loss='binary_crossentropy',
#                     metrics=['accuracy'])
#
#
# train_model.summary()

pre_trained_model = keras.applications.InceptionV3(input_shape=(150, 150,3),
                                                   include_top=False,
                                                   weights='imagenet')

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

model.fit(train_generator,
                validation_data= validation_generator,
                steps_per_epoch=100,
                epochs=3,
                validation_steps=50,)



