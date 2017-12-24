# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 02:16:33 2017

@author: Sahil Manchanda
"""

import pandas as pd
import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical


mushroom_data = pd.read_csv('mushrooms.csv')
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for col in mushroom_data.columns:
    mushroom_data[col] = encoder.fit_transform(mushroom_data[col])
    
    
#from sklearn.preprocessing import OneHotEncoder
#encoder = OneHotEncoder()
#for col in mushroom_data.columns:
#    mushroom_data[col] = encoder.fit_transform(mushroom_data[col])

X = mushroom_data.iloc[:,1:22].values
Y = mushroom_data.iloc[:,0].values
     


from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25,random_state=42)

model = Sequential()
model.add(Dense(40,input_dim = 21,kernel_initializer="normal",activation = 'relu'))
model.add(Dense(10,kernel_initializer="normal",activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(5,kernel_initializer="normal",activation = 'relu'))
model.add(Dense(1,kernel_initializer="normal",activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=200, verbose=2)
print(model.evaluate(x_test,y_test))



