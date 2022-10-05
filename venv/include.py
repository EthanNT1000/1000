import sys, os
import numpy as np


def mean_squared_error(y, t):
  return np.mean((y-t)**2)

class Loss:
    def __int__(self,y,t):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self,y,t):
        self.y = y
        self.t = t
        self.loss = mean_squared_error(self.y,self.t)
        return self.loss

    def backward(self,dout,):
        batch_size = self.t.shape[0]
        dx = 2*dout * (self.y - self.t)/batch_size
        return dx

class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
        it.iternext()

    return grad