import numpy as np

def softmax(a):
    c=np.max(a)
    expa=np.exp(a-c)
    sumexpa=np.sum(expa)
    y=expa/sumexpa
    return y


def gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad
    
def kansu(x):
    return x[0]**2+x[1]**2
a=np.array([-3,4.0])



def gradientdecent(f,initx,lr=0.01,num=100):
    a=initx
    for i in range(num):
        g=gradient(f, a)
        a-=lr*g
       
    return a
    


def crossentro(y,t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))

class simplenet:
    def __init__(self):
        self.w=np.random.randn(2,3)
    def predict(self,x):
        return np.dot(x,self.w)
    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(z)
        loss=crossentro(y,t)
        
        return loss

print(simplenet().w)
x=np.array([0.6,0.9])
p=simplenet().predict(x)
print(p)

t=np.array([0,0,1])
print(simplenet().loss(x,t))

def f(w):
    return simplenet().loss(x,t)

dw=gradient(f,simplenet().w)
print(dw)
    
