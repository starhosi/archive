from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 13 * 3
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

model = Sequential()
model.add(Dense(20,activation='relu', input_shape=(3,)))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.save('./save/save.h5')