from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 13 * 3
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

# x = np.transpose(x)
# y = np.transpose(y)

model = Sequential()

model.add(Dense(20,activation='relu', input_shape=(3,)))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3.훈련
from keras.callbacks import EarlyStopping, TensorBoard
tb = TensorBoard(log_dir='./graph2')

model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])


early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x,y,epochs=1000,batch_size=1,verbose=1,callbacks=[early_stopping,tb])
loss, mae = model.evaluate(x,y,batch_size=1)
print(loss,mae)

x_input = array([[6.5,7.5,8.5],[50,60,70],[70,80,90],[100,110,120]])

y_predict = model.predict(x_input)
print(y_predict)