import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#data preparation using Cirfar10
(independentVar, dependentVar) , _ = tf.keras.datasets.cifar10.load_data()
print(independentVar.shape, dependentVar.shape)
# print("input shape is ", input.shape)
dependentVar = pd.get_dummies(dependentVar.reshape(50000))#[0, 0, 0, 1, 0, 0, 0, 0, 0, 0] -> label 3
print(independentVar.shape, dependentVar.shape)
# #model design
X = tf.keras.layers.Input(shape = [32,32,3])
H = tf.keras.layers.Conv2D(6, kernel_size=5, activation="swish") (X)#based on lenet5 description
H = tf.keras.layers.MaxPool2D() (H)

H = tf.keras.layers.Conv2D(16, kernel_size=5,activation="swish") (H)#based on lenet5 description
H = tf.keras.layers.MaxPool2D() (H)

H = tf.keras.layers.Flatten() (H)
H = tf.keras.layers.Dense(120, activation='swish') (H)
H = tf.keras.layers.Dense(84, activation='swish') (H)
Y = tf.keras.layers.Dense(10, activation = 'softmax') (H)
model = tf.keras.models.Model(X, Y)
model.compile(loss = 'categorical_crossentropy', metrics='accuracy') 
#model train
model.fit(independentVar, dependentVar, epochs=10)
#model using
pred = model.predict(independentVar[0:5])
print("prediction is ")
pd.DataFrame(pred).round(2)
print(pred)
print("answer is ", dependentVar[0:5])
# model.summary()
plt.imshow(independentVar[0])
plt.show()
# plt.imshow(dependentVar[0])