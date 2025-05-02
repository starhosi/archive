
# input model 하나  output model 세개

import numpy as np
x1= np.array([range(1,101),range(101,201),range(301,401)])
x2 = np.array([range(1001,1101),range(1101,1201),range(1301,1401)])

y1 = np.array([range(1,101), range(101,201), range(301,401)]) 
y2 = np.array([range(1001,1101), range(1101,1201), range(1301,1401)]) 
y3 = np.array([range(1,101), range(101,201), range(301,401)]) 

x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, 
                            test_size = 0.2,  shuffle = False) #일단 Train과 Test로 나눠 준 다음

x1_val, x1_test, x2_val, x2_test, y1_val, y1_test = train_test_split(x1_test, x2_test, y1_test,
                            test_size = 0.5, shuffle = False) # Test에서 val&test로 나눠준다

y2_train, y2_test, y3_train, y3_test = train_test_split(y2, y3,
                            test_size = 0.2, shuffle = False) #일단 Train과 Test로 나눠 준 다음

y2_val, y2_test, y3_val, y3_test = train_test_split(y2_test, y3_test,
                            test_size = 0.5, shuffle = False) # Test에서 val&test로 나눠준다

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
dense2 = Dense(3)(dense1)
output1 = Dense(4)(dense2)

# 분기
output_1 = Dense(30)(output1)
output_1 = Dense(3)(output_1)

output_2 = Dense(300)(output1)
output_2 = Dense(5)(output_2)
output_2 = Dense(3)(output_2)

output_3 = Dense(10)(output1)
output_3 = Dense(3)(output_3)

# Sequential 모델과 다르게 함수형 모델을 하단 부분에 모델을 정의
model = Model(inputs=input1, outputs=[output_1,output_2,output_3])
model.summary()
from keras.callbacks import EarlyStopping, TensorBoard
tb = TensorBoard(log_dir='./graph')
#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])       
model.fit(x1_train, [y1_train,y2_train,y3_train] , epochs =100, batch_size=1, validation_data= (x1_val, [y1_val,y2_val,y3_val]),callbacks=[tb])

# #4. 평가
# loss 3개 mse 3개 
aaa = model.evaluate(x1_test, [y1_test,y2_test, y3_test], batch_size=1)

print('mse: ', aaa)

inp1x = [[101,102,103],[104,105,106],[101,102,103]]
inp1x = np.transpose(inp1x)
aaa= model.predict(inp1x, batch_size=1)

y1_predict,y2_predict,y3_predict = model.predict(x1_test,batch_size=1)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
RMSE = (RMSE(y1_test,y1_predict)+RMSE(y1_test,y1_predict)+RMSE(y1_test,y1_predict))/3
print("RMSE: ",RMSE)

from sklearn.metrics import r2_score
r2_y_predict1 = r2_score(y1_test,y1_predict)
r2_y_predict2 = r2_score(y2_test,y2_predict)
r2_y_predict3 = r2_score(y3_test,y3_predict)
r2_y_predict = (r2_y_predict1+r2_y_predict2+r2_y_predict3)/3
print("R2: ",r2_y_predict)