import sys, os
import csv
import numpy as np
from include import *
from collections import OrderedDict

class PCBNet:
  def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01,b1_init=0):
    self.params = {}
    self.params['w1'] = weight_init_std * \
                        np.random.randn(input_size,hidden_size)
    self.params['b1'] = b1_init#np.zeros(hidden_size)

    self.layers = OrderedDict()
    self.layers[' Affine1'] = Affine(self.params['w1'],self.params['b1'])

    self.lastLayer = Loss()

  def predict(self,x):
    for layer in self.layers.values():
      x=layer.forward(x)
      return(x)

  def loss(self,x,t):
    y =self.predict(x)
    return self.lastLayer.forward(y,t)

  def accuracy(self,x,t):
    y = self.predict(x)
    y = np.argmax(y,axis=1)
    if t.ndim != 1: t = np.argmax(t, axis = 1)
    accutacy = np.sum(y == t) / float(x.shape[0])
    return accutacy

  def numerical_gradient(self,x,t):
    loss_w = lambda w: self.loss(x,t)
    grads = {}
    grads['w1'] = numerical_gradient(loss_w,self.params['w1'])
    grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
    return grads


  def gradient(self,x,t):
    self.loss(x,t)

    dout = 1
    dout = self.lastLayer.backward(dout)
    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
      dout = layer.backward(dout)

    grads ={}
    grads['w1'] = self.layers[' Affine1'].dw
    grads['b1'] = 0#self.layers[' Affine1'].db

    return grads

  def gradient_check(self,x,t):

    grand_numerical = self.numerical_gradient(x, t)
    grad_backprop = self.gradient(x, t)
    print("數值微分梯度與反向傳播法梯度的平均誤差 : ")
    for key in grand_numerical.keys():
      diff = np.average(np.abs(grad_backprop[key] - grand_numerical[key]))
      print(key + ": " + str(diff))



