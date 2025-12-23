import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("HousePrice.csv")

X = data[['TV', 'Radio', 'Newspaper']].values.T   # shape (3, m)
Y = data['Sales'].values.reshape(1, -1)           # shape (1, m)

m = X.shape[1]


X_mean = np.mean(X, axis=1, keepdims=True)
X_std = np.std(X, axis=1, keepdims=True)
X_norm = (X - X_mean) / X_std

Y_mean = np.mean(Y)
Y_std = np.std(Y)
Y_norm = (Y - Y_mean) / Y_std


np.random.seed(42)
w = np.random.randn(1, 3) * 0.01
b = np.zeros((1, 1))


def forward(X, w, b):
    return np.dot(w, X) + b

def compute_cost(Y_hat, Y):
    m = Y.shape[1]
    return np.sum((Y_hat - Y)**2) / (2 * m)

def backward(X, Y, Y_hat):
    m = X.shape[1]
    dZ = Y_hat - Y
    dw = np.dot(dZ, X.T) / m
    db = np.sum(dZ) / m
    return dw, db


learning_rate = 0.05
epochs = 2000
losses = []

for i in range(epochs):
    Y_hat = forward(X_norm, w, b)
    cost = compute_cost(Y_hat, Y_norm)
    dw, db = backward(X_norm, Y_norm, Y_hat)

    w -= learning_rate * dw
    b -= learning_rate * db

    losses.append(cost)

    if i % 200 == 0:
        print(f"Epoch {i} | Loss: {cost:.4f}")


print("\nFinal weights:", w)
print("Final bias:", b)


plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
