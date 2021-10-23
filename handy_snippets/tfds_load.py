import tensorflow_datasets as tfds

# 1. TFDS as dictonary

data, metadata = tfds.load('mnist',
                           as_supervised=True,
                           with_info=True)

train, test = data['train'], data['test']

# 2. TFDS as seprate train and test

(train, test), metadata = tfds.load('mnist',
                                    split=['train', 'test'],
                                    as_supervised=True,
                                    with_info=True)

# 3. TFDS train, val1, val2, test


data, metadata = tfds.load('mnist',
                           split=['train[:80%]', 'train[80%:90%]', 'train[90%:]', 'test'],
                           as_supervised=True,
                           with_info=True
                           )

# tf.data.experimental.cardinality(data[0])
# tf.data.experimental.cardinality(data[1])
