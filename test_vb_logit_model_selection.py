# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 23:52:19 2019

@author: M.Havlicek
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import seaborn as sns
from vbm import vbml
from sklearn.metrics import confusion_matrix
from scipy.stats import norm
import pandas as pd
plt.close('all')
# simulate some data

#dimensionality, number of data points
d = 4           #% base dimensionality of X -> y mapping
N = 50         #% number of data points in training set
Ds = np.arange(1,11)
x_range = [-5, 5]
# random weight vector & predictions
w = np.random.randn(d,1)
#% inputs for train/test set
x  = x_range[0] + (x_range[1]-x_range[0])*np.random.rand(N,1)
x_test = np.linspace(x_range[0],x_range[1],300)

X        = np.fliplr(np.vander(np.squeeze(x), d))
X_test   = np.fliplr(np.vander(np.squeeze(x_test), d))
p_y = 1/(1 + np.exp(- X@w))
y = 2 * (np.random.rand(N,1) < p_y) - 1
p_y_test = 1/(1 + np.exp(- X_test@w))
y_test = 2 * (np.random.rand(len(p_y_test),1) < p_y_test)- 1


F = []
pred_loss = []
p_y_vb       = np.zeros((len(y),len(Ds)))
p_y_vb_test = np.zeros((len(y_test),len(Ds)))
for i in range(len(Ds)):
    # train VB regression without ARD
    Xi       = np.fliplr(np.vander(np.squeeze(x), Ds[i]))
    vb_logit = vbml(Xi,y,ard=False,verbose=True)
    vb_logit.fit()
    F.append(vb_logit.lower_bound[-1])
    p_y_vb[:,i]   = vb_logit.predict(Xi)
    # posterior probabilities, and associated choices, based on p > 0.5
    y_vb = 2 * (p_y_vb > 0.5) - 1
    # test
    Xi_test        = np.fliplr(np.vander(np.squeeze(x_test), Ds[i]))
    p_y_vb_test[:,i] = vb_logit.predict(Xi_test)
    # posterior probabilities, and associated choices, based on p > 0.5
    #y_vb_test = 2 * (p_y_vb_test{i} > 0.5) - 1
    #pred_loss.append([np.mean(y_vb != y), np.mean(y_vb_test{i} != y_test)])
    

# model probabilities
p_m = np.exp(F -max(F))/np.sum(np.exp(F-max(F)))



pred_loss = np.array(pred_loss)   
# plot model selection result
fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1.plot(Ds-1, F)
ax1.set_ylabel('VB - lower bound')

ax2 = ax1.twinx()
ax2.plot(Ds-1, p_m,'r')
#ax2.plot(Ds-1, pred_loss[:,0], 'r-',label='Train loss')
#ax2.plot(Ds-1, pred_loss[:,1], 'r--',label='Test loss')
ax2.set_ylabel('Posterior model probability', color='r')
ax2.set_ylim(0,1)
plt.legend()

for tl in ax2.get_yticklabels():
    tl.set_color('r')

plt.xlabel('Polynomial order');


# pick the one with highest prob
index_max = np.argmax(p_m)

p_y_vb_test_max = p_y_vb_test[:,index_max]
# other option is to do Bayesian model averaging
p_y_vb_test_bma = p_y_vb_test@p_m # weighted by posterior model probability


# plot prediction
y1 = y == 1
plt.figure(2) 
plt.plot(x_test, p_y_test, 'k-',label = 'true')
plt.plot(x_test, p_y_vb_test_max,'g--',label='max F')
plt.plot(x_test, p_y_vb_test_bma,'r--',label='bma')
plt.legend()
plt.plot(x[y1], 1 - 0.05 *np.random.rand(len(y[y1]),1), '+',label='y=1')
plt.plot(x[~y1], 0.05 * np.random.rand(len(y[~y1])), 'o',label='y=0')

plt.xlabel('x')
plt.ylabel('p(y=1)')

plt.show()
