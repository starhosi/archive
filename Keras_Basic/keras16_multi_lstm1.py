from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x = x.reshape(x.shape[0],x.shape[1],1)

# LSTM을 여러개 이은것을 구현.
model = Sequential()
# 20의 의미는 LSTM에서 나온 값이 20개
model.add(LSTM(20,activation='relu', return_sequences=True,input_shape=(3,1)))
model.add(LSTM(10,activation='relu', return_sequences=True))
model.add(LSTM(5,activation='relu', return_sequences=True))
model.add(LSTM(6,activation='relu', return_sequences=True))
model.add(LSTM(19,activation='relu', return_sequences=True))
model.add(LSTM(20,activation='relu', return_sequences=True))
model.add(LSTM(5,activation='relu', return_sequences=True))
model.add(LSTM(20,activation='relu', return_sequences=True))
model.add(LSTM(2,activation='relu'))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
model.summary()

#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])       
from keras.callbacks import EarlyStopping, TensorBoard
tb = TensorBoard(log_dir='./graph')

early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x,y,epochs=1000,batch_size=1,verbose=1,callbacks=[early_stopping,tb])
loss, mae = model.evaluate(x,y,batch_size=1)
print(mae)

x_input = array([[6.5,7.5,8.5],[50,60,70],[70,80,90],[100,110,120]])
x_input = x_input.reshape(x_input.shape[0],x_input.shape[1],1)

y_predict = model.predict(x_input)
print(y_predict)
