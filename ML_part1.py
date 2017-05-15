#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:50:37 2017

@author: shawn
"""

import os 
#os.chdir(r'E:\Dropbox\ML part1')
os.chdir(r'/Users/shawn/Dropbox/ML part1')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit

data = pd.read_csv('ex1data1.txt',names=['Footage','HousePrice'])
X = pd.DataFrame(data['Footage'])
y = data['HousePrice']

def loss_func(X, y, theta):
    m, p = X.shape
    #compute mse
    J = sum((X.dot(theta) - y) ** 2.0)/(2.0*m)
    #compute gradient
    grad = np.zeros(p)
    err = X.dot(theta) - y
    grad = (np.tile(err,(p,1)).T * X).sum()*(1.0/m)
    return J, grad

def optimizer_func(X, y, theta, alpha, max_iter):
    for i in range(max_iter):
        J, grad =loss_func(X, y, theta)
        theta = theta - alpha*grad
        print("iteration: " + str(i+1) + " Error: " + str(J) )
    return theta

####################################
start = timeit.default_timer()

max_iter = 1000
theta=[0,0]
alpha = 0.02
#intercept
m, p = X.shape
X['intercept'] = pd.DataFrame(np.ones(m))
cols = X.columns.tolist()
cols = cols[-1:] + cols[0:-1]
X = X[cols]
del cols
m, p = X.shape

theta = optimizer_func(X, y, theta, alpha, max_iter)
print("parameters:\n" + str(theta))

stop = timeit.default_timer()
print("my code GD runing time: " + str(stop - start))
print("GD error:" + str(sum((X.dot(theta) - y) ** 2.0)/(2.0*m)))
####################################
from numpy.linalg import inv
start = timeit.default_timer()

theta_ne = inv(X.T.dot(X)).dot(X.T).dot(y)

stop = timeit.default_timer()
print("my code normal eq runing time: " + str(stop - start))
print("NE error:" + str(sum((X.dot(theta_ne) - y) ** 2.0)/(2.0*m)))
####################################
from sklearn import datasets, linear_model
start = timeit.default_timer()
regr = linear_model.LinearRegression()
regr.fit(X,y)
stop = timeit.default_timer()
yhat= regr.predict(X)
print("sklearn runing time: " + str(stop - start))
print("sklearn error:" + str(sum((yhat - y) ** 2.0)/(2.0*m)))

# plot the scatter and line fit
plt.scatter(X['Footage'],y)
plt.plot(Ｘ['Footage'],X.dot(theta),'-r')
plt.figure
plt.plot(Ｘ['Footage'],yhat,'-g')
del start, stop