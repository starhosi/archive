from numpy import array
from keras.models import Sequential,Model
from keras.layers import Dense, LSTM, Input
# 14 * 3
# x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
#            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
#            [9,10,11],[10,11,12],
#            [20000,30000,40000],[30000,40000,50000],
#            [40000,50000,60000],[100,200,300]])

# y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400])

x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12]])

y = array([4,5,6,7,8,9,10,11,12,13])


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler는 2차원 데이터만 받는다..?
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x = x.reshape(x.shape[0],x.shape[1],1)

from sklearn.model_selection import train_test_split
x_train, x_test,  y_train, y_test = train_test_split(
    x, y, train_size=0.5, random_state=66, shuffle = True)


input1 = Input(shape=(3,1))
LSTM1 = LSTM(20,activation='relu')(input1)
dense1 = Dense(20)(LSTM1)
dense2 = Dense(10)(dense1)
output = Dense(1)(dense2)

model = Model(inputs=input1, outputs=output)

model.compile(optimizer='adam',loss='mse', metrics=['mse'])
model.fit(x_train,y_train,epochs=100,batch_size=1)
loss, mse = model.evaluate(x_test,y_test,batch_size=1)
print(mse)

p_inpx = array([[10,11,12]])
p_inpx = scaler.transform(p_inpx)
p_inpx = p_inpx.reshape(p_inpx.shape[0],p_inpx.shape[1],1)
bbb = model.predict(p_inpx,batch_size=1)
print(bbb)

# y_predict = model.predict(x_test,batch_size=1)
# from sklearn.metrics import r2_score
# r2_y_predict = r2_score(y_test,y_predict)
# print("R2: ",r2_y_predict)