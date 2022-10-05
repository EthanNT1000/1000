import sys, os
import csv
import numpy as np
import matplotlib.pyplot as plt
from include import *
from collections import OrderedDict
from PCBNet import *

X = []
with open('x_data.csv') as text:
  reader = csv.reader(text, quoting=csv.QUOTE_NONNUMERIC)
  for row in reader:
    X.append(row)

T = []
with open('t_data.csv') as text:
  reader = csv.reader(text, quoting=csv.QUOTE_NONNUMERIC)
  for row in reader:
    T.append(row)

x_train = np.array(X[:100])
t_train = np.array(T[:100])
x_test =  np.array(X[100:200])
t_test = np.array(T[100:200])


net = PCBNet(input_size = 3, hidden_size = 1 ,output_size =1,weight_init_std=0.01,b1_init=50)

iters_num = 5000
train_size = x_train.shape[0]
batch_size = 10
learning_rate = 0.0001
train_params_list = []
train_loss_list = []
test_loss_list = []


for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    train_x_batch = x_train[batch_mask]
    train_t_batch = t_train[batch_mask]
    test_x_batch = x_train[batch_mask]
    test_t_batch = t_train[batch_mask]

    grad = net.gradient(train_x_batch,train_t_batch )

    for key in('w1','b1'):
        net.params[key] -= learning_rate *grad[key]

    train_params_list.append(str(net.params))

    train_loss = net.loss(train_x_batch,train_t_batch )
    train_loss_list.append(train_loss)

    test_loss = net.loss(test_x_batch, test_t_batch)
    test_loss_list.append(test_loss)

with open('output_data.csv', 'w', newline='') as csvfile:

    writer = csv.writer(csvfile)
    writer.writerow(["損失函數:"]+ train_loss_list)
    writer.writerow(["權重比:"] + train_params_list)
print('損失函數 : ' + str(train_loss))
print('權重比 : \n' + str(net.params))

plt.plot(train_loss_list, label = "Train Loss", color = 'r')
plt.plot(test_loss_list, label = "Test Loss", color = 'b')
plt.xlabel("iterations")
plt.title("Loss ")
plt.legend()
plt.show()

'''梯度檢查
net.gradient_check(x_batch,t_batch)
'''




