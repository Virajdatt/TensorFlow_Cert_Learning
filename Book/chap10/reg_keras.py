from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np
housing_data = datasets.fetch_california_housing()

x_train, x_test, y_train,  y_test = train_test_split(housing_data.data, housing_data.target)

xt, xv, yt, yv = train_test_split(x_train, y_train)

scaler = StandardScaler()
xt = scaler.fit_transform(xt)
x_test = scaler.transform(x_test)
xv = scaler.transform(xv)

class ViewKeys(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        print("The Learning rate used is ", logs.get('lr'))

model = keras.Sequential([keras.layers.Dense(30, input_shape=xt[1:].shape, activation='relu'),
                          keras.layers.Dense(1)
                        ])
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

optimizer = keras.optimizers.Adam()
lr_metric = get_lr_metric(optimizer)
def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001

def piecewise_constant(boundaries, values):
    boundaries = np.array([0] + boundaries)
    values = np.array(values)
    def piecewise_constant_fn(epoch):
        return values[np.argmax(boundaries > epoch) - 1]
    return piecewise_constant_fn

piecewise_constant_fn = piecewise_constant([5, 15], [0.01, 0.005, 0.001])

lr_scheduler = keras.callbacks.LearningRateScheduler(piecewise_constant_fn)

model.compile(
    optimizer=optimizer,
    metrics=['accuracy', lr_metric],
    loss='mean_squared_error',
    )

#model.compile(loss = "mean_squared_error", optimizer="sgd")
model.fit(xt, yt, epochs=20, validation_data=(xv, yv), callbacks=[lr_scheduler,ViewKeys()])
model.evaluate(x_test, y_test)

model.predict(x_test)