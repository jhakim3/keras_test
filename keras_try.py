from keras.models import Sequential
model=Sequential()
from keras.layers import Dense, Input
model.add(Dense(units=64,activation='relu',input_dim=28*28))
model.add(Dense(units=10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

from keras.datasets import mnist
(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train=X_train.reshape(60000,28*28)
X_test=X_test.reshape(10000,28*28)

model.fit(X_train,y_train,epochs=5,batch_size=32)
loss_and_metrics=model.evaluate(X_test,y_test,batch_size=128)
print(loss_and_metrics)