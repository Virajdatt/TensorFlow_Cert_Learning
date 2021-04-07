import tensorflow as tf

# Custom Activation Function
def my_softplus(z):

    return tf.math.log(tf.exp(z) + 1.0 )

print(my_softplus(10.0))
print(tf.nn.softplus(10.0))

# Output
# tf.Tensor(10.000046, shape=(), dtype=float32)
# tf.Tensor(10.000046, shape=(), dtype=float32)

# Custom Intializers
tf.random.set_seed(42)
def my_glrot(shape, dtype=tf.float32):

    stddev = tf.sqrt(2./(shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype, seed=42)

print(my_glrot((28,28)))
glrot = tf.keras.initializers.GlorotNormal(seed=42)
values = glrot(shape=(28,28),)
print(values)