import tensorflow as tf

t = tf.constant([[1,2,3],
                 [4,5,6]],dtype=tf.float32
                )
print(t)

print(tf.square(t))

# Huber Loss

def hub_loss(ytrue, ypred):

    error = ytrue - ypred
    print("Absolute error is",tf.abs(error))
    is_error_small = tf.abs(error) < 1
    print("abs error", tf.abs(error) < 1)
    squared_error = tf.square(error) / 2
    linear_loss = tf.abs(error) - tf.constant(.5, dtype=tf.float32)
    print("Squared_error is ", squared_error)
    print("Linear error is", linear_loss)
    print("Error returned is",tf.where(is_error_small, squared_error, linear_loss))
t = tf.constant([[1,2,3],
                 [4,5,6]],dtype=tf.float32
                )
s = tf.constant([[1,1,1],
                 [2, 2, 2]
                ],dtype=tf.float32)

hub_loss(t,s)
