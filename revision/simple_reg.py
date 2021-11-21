import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

tf.keras.backend.clear_session()


class Reg_CB(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') < 0.1:
            print('Training Done')
            self.model.stop_training = True

xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype = float)
ys = np.array([100, 150, 200, 250, 300, 350], dtype = float)

SC = StandardScaler()
scaled_y = SC.fit_transform(ys.reshape(-1, 1))

model = tf.keras.Sequential([tf. keras.layers.Dense(units = 1, input_shape = [1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

model.fit(xs, ys/100, epochs=1000, callbacks=Reg_CB())`

print(SC.inverse_transform(model.predict([7.0])))
print(model.predict([7.0]) * 100)

#SC.inverse_transform()
