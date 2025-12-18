import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('tvsales.csv')
#print(data)

plt.scatter(data['TV'],data['Sales'])
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.show()


X = data['TV'].values.reshape(1,-1)
Y = data['Sales'].values.reshape(1,-1)

X_mean = np.mean(X)
X_std = np.std(X)
Y_mean = np.mean(Y)
Y_std = np.std(Y)

X_norm = (X - X_mean)/X_std
Y_norm = (Y - Y_mean)/Y_std


w = np.random.randn(1,1)*0.01
b = np.zeros((1,1))

def forward(X,w,b):
    Y_hat = np.dot(w,X) + b
    return Y_hat



def compute_cost(Y_hat,Y):
    m = Y.shape[1]
    cost = np.sum((Y_hat - Y)**2)/(2*m)
    return cost

def backward(X,Y,Y_hat) :
    m = X.shape[1]
    dZ = Y_hat - Y
    dw = (1/m)*np.dot(dZ,X.T)
    db = (1/m)*np.sum(dZ,axis = 1,keepdims = True)

    return dw,db

#trainning loop
learn_rate = 0.1
iterations = 1000

for i in range(iterations):
    Y_hat = forward(X_norm,w,b)
    cost = compute_cost(Y_hat,Y_norm)
    dw,db = backward(X_norm,Y_norm,Y_hat)

    w = w - learn_rate*dw
    b = b - learn_rate*db

    if i%100 ==0:
        print(f"Iteration {i}, Cost : {cost}")

print("Final w:", w)
print("Final b:", b)
