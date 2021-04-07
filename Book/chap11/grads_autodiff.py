import tensorflow as tf

def f(w1, w2):

    return 3 * w1**2 + 2 * w1 * w2


w1, w2 = 5, 3
eps = 1e-6
print((f(w1 + eps, w2) - f(w1, w2))/ eps)
print((f(w1,w2 +eps ) - f(w1, w2))/ eps)
# 36.000003007075065
# 10.000000003174137



w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])
#[<tf.Tensor: shape=(), dtype=float32, numpy=36.0>, <tf.Tensor: shape=(), dtype=float32, numpy=10.0>]
