"""
A list of tf.data api to be familiar with to use tensorflow much better.
It also is a sort of prerequisite for using tensorflow_datasets.

tf.data API list

apply
map
filter
repeat
shuffle
batch
zip
skip
take
prefetch
"""

import tensorflow as tf
import numpy as np

## Buliding a tf.data.Dataset from numpy for playing
np_array = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
np_array.shape
labels = np.array([0, 0, 1])
d_ds = tf.data.Dataset.from_tensor_slices(np_array)
d_ds.element_spec
lab_tf = tf.data.Dataset.from_tensor_slices(labels)
lab_tf

# Zipping 2 tf datasets
dataset = tf.data.Dataset.zip((d_ds, lab_tf))

for i in dataset.as_numpy_iterator():
    print(i[0], type(i[0]), i[1])
# [1 2 3] <class 'numpy.ndarray'> 0
# [4 5 6] <class 'numpy.ndarray'> 0
# [7 8 9] <class 'numpy.ndarray'> 1

dataset = tf.data.Dataset.from_tensor_slices((np_array, labels))
X, Y = next(dataset.as_numpy_iterator())
#(array([1, 2, 3]), 0)

# Using the maps method
dataset = dataset.map(lambda x, y: (x+1, y))
X, Y = next(dataset.as_numpy_iterator())
# (array([2, 3, 4]), 0)
for i in dataset.as_numpy_iterator():
    print(i[0], type(i[0]), i[1])
# [2 3 4] <class 'numpy.ndarray'> 0
# [5 6 7] <class 'numpy.ndarray'> 0
# [ 8  9 10] <class 'numpy.ndarray'> 1

fdataset = dataset.filter(lambda x, y: x[0]<8 )
for i in fdataset.as_numpy_iterator():
    print(i[0], type(i[0]), i[1])
# [2 3 4] <class 'numpy.ndarray'> 0
# [5 6 7] <class 'numpy.ndarray'> 0

dataset = dataset.repeat(2)
for i in dataset.as_numpy_iterator():
    print(i[0], type(i[0]), i[1])

bdataset = dataset.batch(2)
# list(bdataset.as_numpy_iterator())
for i in bdataset.as_numpy_iterator():
    print(i[0], i[1])

for i in dataset.as_numpy_iterator():
    print(i[0], i[1])

for X, Y in dataset.repeat(4).shuffle(3).batch(2).take(10):
    print(X, Y)


dataset = tf.data.Dataset.from_tensor_slices((np_array, labels))
list(dataset.skip(1).as_numpy_iterator())
# [(array([4, 5, 6]), 0), (array([7, 8, 9]), 1)]
list(dataset.take(1).as_numpy_iterator())
# [(array([1, 2, 3]), 0)]