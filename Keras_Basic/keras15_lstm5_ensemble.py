from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

# input model 두개, output model 두개
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])

y1 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
           [50,60,70],[60,70,80],[70,80,90],[80,90,100],
           [90,100,110],[100,110,120],
           [2,3,4],[3,4,5],[4,5,6]])

y2 = array([40,50,60,70,80,90,100,110,120,130,5,6,7])

x1 = x1.reshape(x1.shape[0],x1.shape[1],1)
x2 = x2.reshape(x2.shape[0],x2.shape[1],1)

from keras.models import Sequential, Model

# Model 1
input1 = Input(shape=(3,1))
LSTM1 = LSTM(20, activation='relu')(input1)
dense11 = Dense(20)(LSTM1)
dense12 = Dense(10)(dense11)
dense13 = Dense(10)(dense12)
output1 = Dense(5)(dense13)

# Model 2
input2 = Input(shape=(3,1))
LSTM2 = LSTM(20, activation='relu')(input2)
dense21 = Dense(20)(LSTM2)
dense22 = Dense(10)(dense21)
dense23 = Dense(10)(dense22)
output2 = Dense(5)(dense23)

from keras.layers.merge import concatenate, Add
# 두방법 모두 사용 가능.
merge1 = concatenate([output1,output2])
# merge1 = Add()([output1,output2])

# Model 3
output_1 = Dense(30)(merge1)
output_1 = Dense(3)(output_1)
output_1 = Dense(1)(output_1)

# Model 4
output_2 = Dense(30)(merge1)
output_2 = Dense(5)(output_2)
output_2 = Dense(1)(output_2)

#3.훈련
model = Model(inputs=[input1,input2], outputs=[output_1,output_2])
model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.fit([x1,x2],[y1,y2],epochs=100,batch_size=1,verbose=1)
aaa = model.evaluate([x1,x2],[y1,y2],batch_size=1)
print(aaa)

x1_input = array([[6.5,7.5,8.5],[6.5,7.5,8.5],[6.5,7.5,8.5],[6.5,7.5,8.5]])
x2_input = array([[6.5,7.5,8.5],[6.5,7.5,8.5],[6.5,7.5,8.5],[6.5,7.5,8.5]])
x1_input = x1_input.reshape(x1_input.shape[0],x1_input.shape[1],1)
x2_input = x2_input.reshape(x2_input.shape[0],x2_input.shape[1],1)

y_predict = model.predict([x1_input,x2_input])
print(y_predict)
