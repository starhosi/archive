from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 5 * 3, 시계열 데이터 : x = 1,2,3 / y = 4 
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7]])
# 5,
y = array([4,5,6,7,8])
# x shaep 5*3 > 5*3*1
x = x.reshape(x.shape[0],x.shape[1],1)

model = Sequential()
# (3, 1)의 '1'은 1개씩 잘라서 계산한다는 뜻
# [1,2,3]의 데이터를  1> 2> 3> 으로 잘라준다.
model.add(LSTM(30,activation='relu', input_shape=(3,1)))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])       
model.fit(x,y,epochs=100,batch_size=1)
loss, mae = model.evaluate(x,y,batch_size=1)
print(loss,mae)

x_input = array([6,7,8])
x_input = x_input.reshape(1,3,1)

y_predict = model.predict(x_input)
print(y_predict)