# 두개의 모델을 concatenate하여 하나의 모델로 만든다.

import numpy as np
#1. 데이터
x1 = np.array([range(1,101),range(101,201),range(301,401)])
x2 = np.array([range(1001,1101),range(1101,1201),range(1301,1401)])
y1 = np.array([range(101,201)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test,  y1_train, y1_test = train_test_split(
    x1, x2, y1, train_size=0.6, random_state=66, shuffle = False)
    
x1_test, x1_val, x2_test, x2_val,  y1_test, y1_val = train_test_split(
    x1_test, x2_test, y1_test, test_size=0.5, random_state=66, shuffle = False)

print(x1_train.shape) 
print(x1_test.shape)
print(x1_val.shape)
print(x2_train.shape) 
print(x2_test.shape)
print(x2_val.shape)
print(y1_train.shape)
print(y1_test.shape)
print(y1_val.shape)

#2. 함수형 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input


# Model 1
input1 = Input(shape=(3,))
dense1 = Dense(5,activation='relu')(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

# Model 2
input2 = Input(shape=(3,))
dense21 = Dense(5,activation='relu')(input2)
dense22 = Dense(2)(dense21)
dense23 = Dense(3)(dense22)
output2 = Dense(1)(dense23)

from keras.layers.merge import concatenate
# axis 주의 !!
merge1 = concatenate([output1,output2])

# Model 3
middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
output = Dense(1)(middle2)

# Sequential 모델과 다르게 함수형 모델을 하단 부분에 모델을 정의
model = Model(inputs=[input1,input2], outputs=output)
model.summary()

#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])
model.fit([x1_train, x2_train], y1_train , epochs =2, batch_size=1, validation_data= ([x1_val,x2_val], y1_val))

#4. 평가
loss, mse = model.evaluate([x1_test, x2_test], y1_test, batch_size=1)
print('mse: ', mse)

inp1x = [[101,102,103,2],[104,105,106,2],[101,102,103,3]]
inp2x = [[101,102,103,2],[104,105,106,2],[101,102,103,2]]
inp1x = np.transpose(inp1x)
inp2x = np.transpose(inp2x)
aaa = model.predict([inp1x,inp2x], batch_size=1)
print(aaa)

y_predict = model.predict([x1_test,x2_test],batch_size=1)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y1_test,y_predict))
print("RMSE: ",RMSE(y1_test,y_predict))

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y1_test,y_predict)
print("R2: ",r2_y_predict)