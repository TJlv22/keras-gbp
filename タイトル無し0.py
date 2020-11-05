from keras import models
from keras.layers import LSTM,Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils

data=pd.read_csv("1440.csv")
data=pd.DataFrame(data[["realreturn","future_direction"]])

"""p=2
for lag in range(1, p+1) :
    col="lag_{}".format(lag)
    data[col]=data["realreturn"].shift(lag)
    """
    
direction=np.array(data["future_direction"])
nd=np.where(direction==1,1,0)
data["future_direction"]=nd

#create data
datax=[]
datay=[]
p=20

for i in range(len(data)-p) :
    datax.append(data["realreturn"][i:i+p])
    datay.append(data["future_direction"][i+p])
    
datax=np.array(datax)
datay=np.array(datay)
datay=np_utils.to_categorical(datay)

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
ytrain1=scaly.fit_transform(ytrain.reshape(len(ytrain),2))
ytest1=scaly.fit_transform(ytest.reshape(len(ytest),2))

#change shape of data
xtrain1=np.reshape(xtrain1,(xtrain1.shape[0],1,xtrain1.shape[1]))
xtest1=np.reshape(xtest1,(xtest1.shape[0],1,xtest1.shape[1]))

#create model
model=models.Sequential()
model.add(LSTM(128,activation="tanh",input_shape=(1,p)))
#model.add(Dense(128,activation="relu"))
model.add(Dense(2,activation="sigmoid"))
model.compile(loss="categorical_crossentropy",optimizer="adam")
#model.compile(loss="mean_squared_error", optimizer="adam")

#learn and plot
result=model.fit(xtrain1,ytrain1,batch_size=100,epochs=100)
yp=model.predict(xtest1)
yp=scaly.inverse_transform(yp).flatten()
plt.figure(figsize=(10,6))
plt.plot(yp,label="predict")
plt.plot(ytest,label="price")
plt.legend()


















