# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:51:07 2019

@author: M.Havlicek
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from vbm import vbmr

plt.close('all')
# simulate some data
N = 30 # number of sample points
N_ex = 50

d = 7 # lenght
d_ex = 9
features   = 7


# simulate data
x0_train   = np.random.uniform(0,d,N)
x0_test    = np.linspace(0, d, N)
x0_test_ex = np.linspace(0, d_ex, N_ex)

# get polynomial basis
X_test_ex         = np.fliplr(np.vander(x0_test_ex, features))
X_std             = np.std(X_test_ex[:,1:],axis=0)
X_test_ex[:,1:]   = X_test_ex[:,1:]/X_std
X_train           = np.fliplr(np.vander(x0_train, features))
X_train[:,1:]     = X_train[:,1:]/X_std
X_test            = np.fliplr(np.vander(x0_test, features))
X_test[:,1:]      = X_test[:,1:]/X_std


noise     = np.random.randn(N)
w_true    = [5,-3,-4,6]   # use only 4 basis to generate targets
y_train   = X_train[:,:len(w_true)].dot(w_true) + noise
y_test    = X_test[:,:len(w_true)].dot(w_true) 
y_test_ex = X_test_ex[:,:len(w_true)].dot(w_true) 

# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_train, y_train)
y_ml_train = regr.predict(X_train)
# Make predictions using the testing set
y_ml_test  = regr.predict(X_test)
y_ml_test_ex  = regr.predict(X_test_ex)

# train VB regression without ARD
vb_reg = vbmr(X_train,y_train,ard=False,verbose=True)
vb_reg.fit()
y_vb_train, y_vb_sd_train     = vb_reg.predict_dist(X_train)
y_vb_test, y_vb_sd_test        = vb_reg.predict_dist(X_test)
y_vb_test_ex, y_vb_sd_test_ex  = vb_reg.predict_dist(X_test_ex)

# train VB regression with ARD
vb_reg_ard = vbmr(X_train,y_train,ard=True,verbose=True)
vb_reg_ard.fit()
y_vb_ard_train, y_vb_ard_sd_train      = vb_reg_ard.predict_dist(X_train)
y_vb_ard_test, y_vb_ard_sd_test        = vb_reg_ard.predict_dist(X_test)
y_vb_ard_test_ex, y_vb_ard_sd_test_ex  = vb_reg_ard.predict_dist(X_test_ex)


plt.figure(1)
plt.plot(x0_train, y_train, 'r.',label='Train')
plt.plot(x0_test, y_test, 'k',label='True')
plt.plot(x0_test, y_vb_test, 'y',label='VB')
# plot confidence limits
plt.fill_between(x0_test, y_vb_test + 1.96*y_vb_sd_test, 
                 y_vb_test -1.96*y_vb_sd_test, facecolor='yellow', alpha=.1)
plt.plot(x0_test, y_vb_ard_test, 'g',label='VB-ARD')
plt.fill_between(x0_test, y_vb_ard_test + 1.96*y_vb_ard_sd_test, 
                 y_vb_ard_test -1.96*y_vb_ard_sd_test, facecolor='green', alpha=.1) 

plt.plot(x0_test, y_ml_test, 'b',label='ML')
plt.legend()
plt.show()


plt.figure(3)
plt.plot(x0_train, y_train, 'r.',label='Train')
plt.plot(x0_test_ex, y_test_ex, 'k',label='True')
plt.plot(x0_test_ex, y_ml_test_ex, 'b--',label='ML')
plt.plot(x0_test_ex, y_vb_test_ex, 'y--',label='VB')
plt.plot(x0_test_ex, y_vb_ard_test_ex, 'g--',label='VB-ARD')
plt.fill_between(x0_test_ex, y_vb_test_ex + 1.96*y_vb_sd_test_ex, 
                             y_vb_test_ex -1.96*y_vb_sd_test_ex, facecolor='yellow', alpha=.2)
plt.ylim(-10,10)
plt.legend()
plt.show()
# Cross-validation
# print training set and cross-validated MSE
print('MSEs:       training set     test set    test set (Ext)\n')
print('ML          {0:7.5f}         {1:7.5f}    {2:7.5f}\n'.format(
        np.mean((y_train - y_ml_train)**2), np.mean((y_test - y_ml_test)**2), np.mean((y_test_ex - y_ml_test_ex)**2)))
print('VB          {0:7.5f}         {1:7.5f}    {2:7.5f}\n'.format(
        np.mean((y_train - y_vb_train)**2), np.mean((y_test - y_vb_test)**2), np.mean((y_test_ex - y_vb_test_ex)**2)))
print('VB-ARD      {0:7.5f}         {1:7.5f}    {2:7.5f}\n'.format(
        np.mean((y_train - y_vb_ard_train)**2), np.mean((y_test - y_vb_ard_test)**2), np.mean((y_test_ex - y_vb_ard_test_ex)**2)))


