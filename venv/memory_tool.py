import sys, os
import csv
import numpy as np

num = 93
A =np.zeros(shape=(num*7,num))
r = np.array([1,2,3,5,8,13,21])
B =[]
for i in range(num):
  k = r+i
  for j in range(6):
    A[k[j],i] = i+1

with open('english.csv', 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  for z in range(A.shape[0]):
    for g in range(A.shape[1]):
      if A[z,g] != 0:
        B.append(A[z,g])

    writer.writerow(B)
    list.clear(B)



