from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import pydot
housing_data = datasets.fetch_california_housing()

x_train_f, x_test, y_train_f,  y_test = train_test_split(housing_data.data, housing_data.target)

xtrain, xvalid, ytrain, yvalid = train_test_split(x_train_f, y_train_f)

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
x_test = scaler.transform(x_test)
xvalid = scaler.transform(xvalid)

xtrain_A, xtrain_B = xtrain[:, :5], xtrain[:, 2:]
xvalid_A, xvalid_B = xvalid[:, :5], xvalid[:, 2:]
x_test_A, x_test_B = x_test[:, :5], x_test[:, 2:]


inputA = keras.Input(shape=[5], name="InputA")
inputB = keras.Input(shape=[6], name="InputB")
hidden1 = keras.layers.Dense(30, activation='relu', name='Hidden1' )(inputB)
hidden2 = keras.layers.Dense(30, activation='relu', name='Hidden2' )(hidden1)
concat = keras.layers.concatenate([inputA, hidden2], name='Concat')
output = keras.layers.Dense(1, name='Output')(concat)
aux_output = keras.layers.Dense(1, name='Aux_Output')(hidden2)
model = keras.Model(inputs=[inputA, inputB], outputs = [output, aux_output])

model.compile(loss = ['mse', 'mse'], optimizer="sgd")
model.summary()

model.fit( (xtrain_A, xtrain_B), [ytrain, ytrain], epochs=20,
            validation_data=((xvalid_A, xvalid_B),[yvalid, yvalid] ))