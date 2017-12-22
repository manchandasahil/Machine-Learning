# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:27:46 2017

@author: Sahil Manchanda
"""
import pandas as pd
import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

wine_data = pd.read_csv('wine.data',names = ['Class','Alcohol','Malic acid','Ash','Alcalinity','Magnesium','phenols','Flavanoids'
 	,'Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline'], index_col = False)
#print(wine_data.describe())
y = wine_data['Class'].values


from keras.utils import np_utils
y = np_utils.to_categorical(y)[:,1:4]
print(y)
del(wine_data['Class'])
wine_data = wine_data.values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(wine_data,y,test_size = 0.25,random_state=42)

model = Sequential()
model.add(Dense(40,input_dim = 13,kernel_initializer="normal",activation = 'relu'))
model.add(Dense(10,kernel_initializer="normal",activation = 'sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(5,kernel_initializer="normal",activation = 'sigmoid'))
model.add(Dense(3,kernel_initializer="normal",activation = 'softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=1600, verbose=1)
y_pred = model.predict(x_test)
print(model.evaluate(x_test,y_test))
