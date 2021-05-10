import tensorflow as tf
import numpy as np

data = tf.data.Dataset.range(10)
list(data.as_numpy_iterator())
#  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

windowed_data = data.window(5, shift=1, drop_remainder=True)
for window_data in windowed_data:
    for val in window_data:
        print(val.numpy(), end=" ")
    print()

dataset = windowed_data.flat_map(lambda window: window.batch(5))
for val in dataset:
    print(val.numpy())

dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
for x, y in dataset:
    print(x.numpy(), y.numpy())

dataset = dataset.shuffle(buffer_size=10)
for x, y in dataset:
    print(x.numpy(), y.numpy())


dataset = dataset.batch(2).prefetch(1)
for x,y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())




# Pieceing everything together

ndataset = tf.data.Dataset.range(10)
#ndataset = tf.expand_dims(ndataset, axis=-1)
ndataset = ndataset.window(5, shift=1, drop_remainder=True)
ndataset = ndataset.flat_map(lambda window: window.batch(5))
ndataset = ndataset.map(lambda window: (window[:-1], window[-1:]))
ndataset = ndataset.shuffle(buffer_size=10)
ndataset = ndataset.batch(3).prefetch(1)

for x, y in ndataset:
    print("x = ", x.numpy())
    print("y = ", y.numpy())