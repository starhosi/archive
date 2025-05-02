from numpy import array
import numpy as np

def split_sequence2(sequence,n_steps):
    X,y = list(),list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence):
            break
        print(sequence)
        seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix-1,-1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

in_seq1 = array([10,20,30,40,50,60,70,80,90,100])
in_seq2 = array([15,25,35,45,55,65,75,85,95,105])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

print(in_seq1.shape)
print(in_seq2.shape)
print(out_seq.shape)

in_seq1 = in_seq1.reshape(len(in_seq1),1)
in_seq2 = in_seq2.reshape(len(in_seq2),1)
out_seq = out_seq.reshape(len(out_seq),1)

print(in_seq1.shape)
print(in_seq2.shape)
print(out_seq.shape)

from numpy import hstack
dataset = hstack((in_seq1, in_seq2,out_seq))
n_steps = 3

x,y = split_sequence2(dataset,n_steps)

for i in range(len(x)):
    print(x[i],y[i])
    
print(x.shape)
print(y.shape)

from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()

model.add(LSTM(20, input_shape=(3,2)))
model.add(Dense(units=5))
model.add(Dense(units=4))
model.add(Dense(units=3))
model.add(Dense(units=1))
model.summary()

model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])
model.fit(x,y,epochs=100,batch_size=1,verbose=1)
loss, mae = model.evaluate(x,y,batch_size=1)
print(mae)

x_input = array([[90,95],[100,105],[110,115]])
print(x_input.shape)
x_input = x_input.reshape(1,x_input.shape[0],x_input.shape[1])
y_predict = model.predict(x_input)
print(y_predict)