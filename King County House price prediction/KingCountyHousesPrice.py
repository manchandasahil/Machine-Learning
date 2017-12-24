# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 00:35:38 2017

@author: Sahil Manchanda
"""
import pandas as pd
import numpy as np
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential

house_data = pd.read_csv('kc-house-data.csv')

#list1 = house_data.axes[1].tolist()
#print(list1)
Y = house_data['price'].values
columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
X = house_data[columns].values
              
zipcodes = pd.get_dummies(house_data['zipcode']).values
condition = pd.get_dummies(house_data['condition']).values
grade = pd.get_dummies(house_data['grade']).values
X = np.concatenate((X,zipcodes),axis = 1)
X = np.concatenate((X,condition),axis = 1)
X = np.concatenate((X,grade),axis = 1)

Y = Y/1000
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25,random_state=42)

model = Sequential()
model.add(Dense(50,input_dim = 103,kernel_initializer="normal",activation = 'relu'))
model.add(Dense(20,kernel_initializer="normal",activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(5,kernel_initializer="normal",activation = 'relu'))
model.add(Dense(1,kernel_initializer="normal"))

model.compile(loss='mae',
              optimizer='adam',
              metrics=['mae'])

model.fit(x_train,y_train,epochs=200, verbose=1)
print(model.evaluate(x_test,y_test))

                          