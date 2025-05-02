from numpy import array

def split_sequence(sequence,n_steps):
    X,y = list(),list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix],sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

dataset = [1,2,3,4,5,6,7,8,9,10]
n_steps = 3

x,y = split_sequence(dataset,n_steps)

x = x.reshape(x.shape[0],x.shape[1],1)

from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()

model.add(LSTM(20, input_shape=(n_steps,1)))
model.add(Dense(units=5))
model.add(Dense(units=4))
model.add(Dense(units=3))
model.add(Dense(units=1))
model.summary()

model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])
model.fit(x,y,epochs=100,batch_size=1,verbose=1)
loss, mae = model.evaluate(x,y,batch_size=1)
print(mae)

x_input = array([80,90,100])
y_predict = model.predict(x_input)