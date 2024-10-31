from keras.models import Sequential
from keras.layers import Dense
import numpy

#随机梯度下降的随机种子
seed = 7
numpy.random.seed(seed)

#加载数据
dataset = numpy.loadtxt("FeaturesWithLabels-1.csv", delimiter=",")


#将数据分为输入和标签
X = dataset[:,1:]
Y = dataset[:,0:1]

print(X)
print(Y)

#创建模型
model = Sequential()
model.add(Dense(222, activation='relu')) 
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#训练模型
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy']) # Fit the model
model.fit(X, Y)

#评估模型
scores = model.evaluate(X, Y, batch_size=3)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
