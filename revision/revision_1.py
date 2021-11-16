import tensorflow as tf
import tensorflow_datasets as tfds

#tf.keras.backend.clear_session()

data, metadata = tfds.load('mnist', with_info='True', as_supervised=True)

train, test = data['train'], data['test']


train = train.batch(32)

for x, y in train.take(1):
    print(x.numpy().shape, y.numpy().shape)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape = (28, 28, 1), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(63, (2,2), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam',
              metrics=['accuracy'],
              loss='sparse_categorical_crossentropy')

model.fit(train, epochs=2)

test = test.batch(32)

model.evaluate(test)
