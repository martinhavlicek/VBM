# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 10:02:49 2020

@author: M.Havlicek
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from vbm import vbmr
#from vb_glm import vb_linear_predict
from scipy.stats import norm
plt.close('all')
# simulate some data
N = 100 # number of sample points

d = 5

X         = np.concatenate((np.ones((N, 1)), np.random.randn(N,d-1)),axis=1)
noise     = np.random.randn(N,1)
w_true    = np.array([[1,2,3,4,5]]).T
y         = X @ w_true + noise


# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X,y)
y_ml = regr.predict(X)


# train VB regression without ARD
vb_reg = vbmr(X,y,ard=True,verbose=True)
vb_reg.fit()
y_vb, y_vb_sd = vb_reg.predict_dist(X)


# train VB regression with ARD
vb_reg_ard = vbmr(X,y,ard=True,verbose=True)
vb_reg_ard.fit()
y_vb_ard, y_vb_ard_sd = vb_reg_ard.predict_dist(X)

fig = plt.figure(1)
fig.set_size_inches(6.4, 8)
w_axis = np.linspace(-2, 8, 400)
w_reg  = np.concatenate((regr.intercept_, np.squeeze(regr.coef_[:,1:])),axis=0)
w_vb, C_vb = vb_reg.w, vb_reg.C 
w_vb_ard, C_vb_ard = vb_reg_ard.w, vb_reg_ard.C 
# Plot weight posteriors
for i in range(w_vb.shape[0]):
    plt.subplot(w_vb.shape[0], 1, i+1)
    plt.plot(w_axis, norm.pdf(w_axis,w_vb[i],np.sqrt(C_vb[i,i])))
    plt.axvline(x=w_reg[i], linestyle='--', color='r')
    try:
        plt.axvline(x=w_true[i], linestyle='--', color='k')
    except:
        plt.axvline(x=0, linestyle='--', color='k')
    plt.title('W[{}]'.format(i))
    

plt.tight_layout()
plt.show()


fig = plt.figure(2)
ax = fig.add_subplot(111)
plt.plot([-25,25],[-25,25],'k--')
plt.errorbar(y,y_vb,yerr=y_vb_sd*1.96,linestyle='None', marker='o',label = 'VB')
plt.plot(y,y_ml,'r.',label = 'ML')
plt.xlim(-25,25)
plt.ylim(-25,25)
plt.legend()
ax.set_aspect('equal', adjustable='box')

plt.xlabel("y")
plt.ylabel("y-predicted")


