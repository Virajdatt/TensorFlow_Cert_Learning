import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

raw_data = pd.read_csv("./Data/daily-min-temperatures.csv")
#raw_data.tail()

time_steps = np.array(list(range(len(raw_data))))
full_values = raw_data['Temp'].values


def plot_series(ts, values, title='Default Title',fig=0):
    plt.figure(fig)
    plt.plot(ts, values, '-')
    plt.grid(True)
    plt.title(title)



#plt.plot(time_steps, full_values, '-')

time_split = 2500
x_train = full_values[:time_split]
time_train = time_steps[:time_split]
x_valid = full_values[time_split:]
time_val = time_steps[time_split:]

plot_series( time_train, x_train,  'Training Data',1)
plot_series( time_val, x_valid, 'Val Data',2)

# Some HyperParams
window_size = 30
batch_size = 32
shuffle_buffer = 1000
tf.random.set_seed(42)
np.random.seed(42)
EPOCHS = 100
keras.backend.clear_session()

def window_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


train_set = window_dataset(x_train, window_size, batch_size, shuffle_buffer)

# Model 1 RNN

model = keras.Sequential([
    keras.layers.SimpleRNN(40, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(40),
    keras.layers.Dense(1)
])


model.compile(loss=keras.losses.Huber(),
              optimizer=keras.optimizers.Adam(1e-8, ),
              metrics=['mae']
              )


LRS = keras.callbacks.LearningRateScheduler(lambda epoch:
                                            1e-8 * 10 ** (epoch/20)
                                            )

history = model.fit(train_set,
          callbacks=LRS,
          epochs=EPOCHS
          )

# plt.semilogx(history.history['lr'],
#              history.history['loss'],
#              )
# plt.axis([1e-8, 1e-4, 0, 60])

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

rnn_forecast = model_forecast(model, full_values[..., np.newaxis], window_size)
rnn_forecast.shape
rnn_forecast = rnn_forecast[time_split - window_size:-1, -1, 0]

plot_series(time_val, x_valid)
plot_series(time_val,rnn_forecast)


time_split
full_values.shape