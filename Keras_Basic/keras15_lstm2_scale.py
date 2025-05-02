from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 13 * 3 (x=1,2,3 / y=4) or (x=20,30,40 / y=50)
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x = x.reshape(x.shape[0],x.shape[1],1)

model = Sequential()
# (3, 1)의 '1'은 1개씩 잘라서 계산한다는 뜻
# [1,2,3]의 데이터를  1> 2> 3> 으로 잘라준다.
model.add(LSTM(20,activation='relu', input_shape=(3,1)))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])       
model.fit(x,y,epochs=100,batch_size=1)
loss, mae = model.evaluate(x,y,batch_size=1)
print(loss,mae)

x_input = array([[6.5,7.5,8.5],[50,60,70],[70,80,90],[100,110,120]])
x_input = x_input.reshape(x_input.shape[0],x_input.shape[1],1)

y_predict = model.predict(x_input)
print(y_predict)