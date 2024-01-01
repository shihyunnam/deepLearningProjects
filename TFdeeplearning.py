#nonflattened data, flattened data, Conv2D, MaxPool2D
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#mnist : handwritten imageset (60000,28, 28)
#cifar10 : 10 different image set for classification

#Deep learning model -> (60000, 784)input -> (60000, 84)hidden layers->(60000,10)output
#중간에 가중치가 높다는것은 해당 픽셀값이 매우 중요하다는것
#hidden layer -> 이미지들이 0에서 9까지어느숫자인지 판단하기 위해서 가장 좋은 특징 84개를 추출
#Convolution -> 특정한 패턴의 특징이 어디서 나타나는지 판단하는도구 + FILTERING

#흑백
#-----------------------1.Using Non flattened data-------------------
    #data preparation
(mnist_x, mnist_y) , _ = tf.keras.datasets.mnist.load_data()
print(mnist_x.shape, mnist_y.shape)
input = mnist_x.reshape(60000, 784)
output = pd.get_dummies(mnist_y)#[0, 0, 0, 1, 0, 0, 0, 0, 0, 0] -> label 3
print(input.shape, output.shape)
#model design
X = tf.keras.layers.Input(shape = [784])
hidden_layer = tf.keras.layers.Dense(84, activation='swish')(X)
Y = tf.keras.layers.Dense(10,activation='softmax') (hidden_layer)
model = tf.keras.models.Model(X, Y)
model.compile(loss = 'categorical_crossentropy', metrics = "accuracy")
#model train
model.fit(input,output,epochs=10)
#Using model
pred = model.predict(input[0:5])
print("prediction is ")
pd.DataFrame(pred.round(2))
print(pred)
print("answer is ", output[0:5])


#2.-----------------------Using flattened data-------------------
    #data preparation
(mnist_x, mnist_y) , _ = tf.keras.datasets.mnist.load_data()
print(mnist_x.shape, mnist_y.shape)
input = mnist_x
print("input shape is ", input.shape)
output = pd.get_dummies(mnist_y)#[0, 0, 0, 1, 0, 0, 0, 0, 0, 0] -> label 3
print(input.shape, output.shape)
#model design
X = tf.keras.layers.Input(shape = [28,28])
H = tf.keras.layers.Flatten() (X) # -> 784로변환
hidden_layer = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10,activation='softmax') (hidden_layer)
model = tf.keras.models.Model(X, Y)
model.compile(loss = 'categorical_crossentropy', metrics = "accuracy")
#model train
model.fit(input,output,epochs=5)
#Using model
pred = model.predict(input[0:5])
print("prediction is ")
pd.DataFrame(pred.round(2))
print(pred)
print("answer is ", output[0:5])

#3.-----------------------Using Convolutional layers------------------- uses 3d array to show spational dimension (channel -> RGB)
    #data preparation
(mnist_x, mnist_y) , _ = tf.keras.datasets.mnist.load_data()
print(mnist_x.shape, mnist_y.shape)
input = mnist_x.reshape(60000,28,28,1)
print("input shape is ", input.shape)
output = pd.get_dummies(mnist_y)#[0, 0, 0, 1, 0, 0, 0, 0, 0, 0] -> label 3
print(input.shape, output.shape)
#model design
X = tf.keras.layers.Input(shape = [28,28,1])
H = tf.keras.layers.Conv2D(3, kernel_size=5,activation="swish") (X)#3filtering, kernel = 5*5 ,이미지의 각 부분을 3가지 다른 방식으로 분석 (ex: 색) 
H = tf.keras.layers.Conv2D(6, kernel_size=5,activation="swish") (H)#6filtering, kernel = 5*5
H = tf.keras.layers.Flatten() (H)
H = tf.keras.layers.Dense(84, activation='swish') (H)
Y = tf.keras.layers.Dense(10, activation = 'softmax') (H)
model = tf.keras.models.Model(X, Y)
model.compile(loss = 'categorical_crossentropy', metrics='accuracy') 
#model train
model.fit(input, output, epochs=10)
#model using
pred = model.predict(input[0:5])
print("prediction is ")
pd.DataFrame(pred).round(2)
print(pred)
print("answer is ", output[0:5])
model.summary()




#컬러
# (cifar_x, cifar_y) , _ = tf.keras.datasets.cifar10.load_data()
# print(cifar_x.shape, cifar_y.shape)




# plt.imshow(mnist_x[0], cmap = 'gray')
# plt.show()

#2차원 이미지를 1차원긴배열로 전환


