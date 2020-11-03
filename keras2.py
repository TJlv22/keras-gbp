import tensorflow as tf
from tensorflow import keras
import pandas as pd 
import numpy as np
from keras import models
from keras.layers import LSTM,Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("1440.csv")

#create data
datax=[]
datay=[]
p=100

for i in range(len(data)-p) :
    datax.append(data["close"][i:i+p])
    datay.append(data["close"][i+p])
    
datax=np.array(datax)
datay=np.array(datay)

#split data
l=int(len(datax)*0.8)
xtrain=datax[:l]
xtest=datax[l:]
ytrain=datay[:l]
ytest=datay[l:]


#normalize data
scal=StandardScaler()
scaly=StandardScaler()
xtrain1=scal.fit_transform(xtrain)
xtest1=scal.fit_transform(xtest)
ytrain1=scaly.fit_transform(ytrain.reshape(len(ytrain),1))
ytest1=scaly.fit_transform(ytest.reshape(len(ytest),1))

#change shape of data
xtrain1=np.reshape(xtrain1,(xtrain1.shape[0],1,xtrain1.shape[1]))
xtest1=np.reshape(xtest1,(xtest1.shape[0],1,xtest1.shape[1]))

#create model
model=models.Sequential()
model.add(LSTM(128,activation="tanh",input_shape=(1,p)))
#model.add(Dense(128,activation="relu"))
model.add(Dense(1,activation="linear"))
#model.compile(loss="categorical_crossentropy",optimizer="adam")
model.compile(loss="mean_squared_error", optimizer="adam")

#learn and plot
result=model.fit(xtrain1,ytrain1,batch_size=100,epochs=500)
yp=model.predict(xtest1)
yp=scaly.inverse_transform(yp).flatten()
plt.figure(figsize=(10,6))
plt.plot(yp,label="predict")
plt.plot(ytest,label="price")
plt.legend()