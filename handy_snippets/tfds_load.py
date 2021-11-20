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


# 3. Load the data as train, validation , test
(data, metadata) = tfds.load('mnist',split=['train[:90%]', 'train[90%:]', 'test'],as_supervised=True,with_info=True)

train_data = data[0]
valid_data = data[1]
test_data = data[2]
