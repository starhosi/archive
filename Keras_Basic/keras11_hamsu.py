###
# 함수형 모델 구축
###
import numpy as np
#1. 데이터
x= np.array([range(1,101),range(101,201),range(301,401)])
y2= np.array([range(101,201)])
y = np.array(range(101,201))

x = np.transpose(x)
y = np.transpose(y)

from sklearn.model_selection import train_test_split
x_train, x_test,  y_train, y_test = train_test_split(
    x, y, train_size=0.6, random_state=66, shuffle = False)

x_test, x_val,  y_test, y_val = train_test_split(
    x_test, y_test, test_size=0.5, random_state=66, shuffle = False)

print(x.shape) 
print(y.shape)

from keras.models import Sequential, Model
from keras.layers import Dense, Input

# input 3은 feature 3과 같은 의미, 열
input1 = Input(shape=(3,))
dense1 = Dense(5,activation='relu')(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output = Dense(1)(dense3)

# Sequential 모델과 다르게 함수형 모델을 하단 부분에 모델을 정의
model = Model(inputs=input1, outputs=output)
model.summary()

#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs =100, batch_size=1, validation_data= (x_val, y_val))

#4. 평가
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('mse: ', mse)

x_prd = np.array([[101,102,103],[104,105,106],[101,102,103]])
x_prd = np.transpose(x_prd)
aaa= model.predict(x_prd, batch_size=1)
print(aaa)

y_predict = model.predict(x_test,batch_size=1)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE: ",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test,y_predict)
print("R2: ",r2_y_predict)