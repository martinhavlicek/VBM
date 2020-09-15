# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:16:31 2019

@author: M.Havlicek
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import seaborn as sns
from vbm import vbml
from sklearn.metrics import confusion_matrix

import pandas as pd
plt.close('all')
# simulate some data

#dimensionality, number of data points
d = 3           #% base dimensionality of X -> y mapping
d_extra = 1    #% additional, uninformative dimensions
N = 100         #% number of data points in training set
N_cv = 100      #% number of data points in test set
X_scale = 5

# random weight vector & predictions
w = np.random.randn(d,1)
# inputs for train/test set
X = np.concatenate((np.ones((N, 1)), X_scale*(np.random.rand(N,1)-0.5)),axis=1)
X = np.concatenate((X,X_scale*(np.random.rand(N, 1)-0.5)-(w[0]+X[:,[1]]*w[1])/w[2]),axis=1)
X_cv = np.concatenate((np.ones((N, 1)), X_scale*(np.random.rand(N,1)-0.5)),axis=1)
X_cv = np.concatenate((X_cv,X_scale*(np.random.rand(N,1)-0.5) - \
                         (w[0]+X_cv[:,[1]]*w[1])/w[2]),axis=1)

p_y = 1/(1 + np.exp(- X@w))
y = 2 * (np.random.rand(N,1) < p_y) - 1
y1 = (y == 1)
y_cv = 2 * (np.random.rand(N_cv,1) < 1/(1 + np.exp(-X_cv[:,:d] @ w))) - 1


# SVM
svc = SVC(kernel='linear', probability=True)
svc.fit(X, y)
y_svc = svc.predict(X)
# predict probabilities
svc_probs = svc.predict_proba(X)
# keep probabilities for the positive outcome only
svc_probs = svc_probs[:, 1]
# calculate scores



# train VB regression without ARD
vb_logit = vbml(X,y,ard=False,verbose=True)
vb_logit.fit()
p_y_vb = vb_logit.predict(X)
# posterior probabilities, and associated choices, based on p > 0.5
y_vb = 2 * (p_y_vb > 0.5) - 1


# train VB regression without ARD
vb_logit_ard = vbml(X,y,ard=True,verbose=True)
vb_logit_ard.fit()
p_y_vb_ard = vb_logit_ard.predict(X)
# posterior probabilities, and associated choices, based on p > 0.5
y_vb_ard = 2 * (p_y_vb_ard > 0.5) - 1

# cross-validation:
p_y_vb_cv = vb_logit.predict(X_cv)
y_vb_cv = 2 * (p_y_vb_cv > 0.5) - 1
p_y_vb_ard_cv = vb_logit_ard.predict(X_cv)
y_vb_ard_cv = 2 * (p_y_vb_ard_cv > 0.5) - 1

# plot data and discriminating hyperplane

w_vb     = vb_logit.w
w_vb_ard = vb_logit_ard.w
w_svc    = np.squeeze(svc.coef_)
plt.figure(1)  
plt.plot(X[np.ix_(np.squeeze(~y1),[1])], X[np.ix_(np.squeeze(~y1),[2])],'b.')
plt.plot(X[np.ix_(np.squeeze(y1),[1])], X[np.ix_(np.squeeze(y1),[2])], 'r.')
xlims = np.array([-3, 3])

plt.plot(xlims, -(w[0] + w[1] * xlims) / w[2], 'k-', label='True')
plt.plot(xlims, -(w_vb[0] + w_vb[1] * xlims) / w_vb[2], 'r-', label='VB')
plt.plot(xlims, -(w_svc[0] + w_svc[1] * xlims) / w_svc[2], 'c-', label='SVM-linear')
plt.plot(xlims, -(w_vb_ard[0] + w_vb_ard[1] * xlims) / w_vb_ard[2], 'b-', label='VB-ARD')

plt.legend()
plt.show()


#print training set and cross-validated MAE
print('MAEs:       training set     test set\n')
print('VB          {0:7.5f}          {1:7.5f}\n'\
        .format(np.mean(abs(0.5*(y - y_vb))), np.mean(abs(0.5*(y_cv - y_vb_cv)))))
print('VB-ard      {0:7.5f}          {1:7.5f}\n'\
        .format(np.mean(abs(0.5*(y - y_vb_ard))), np.mean(abs(0.5*(y_cv - y_vb_ard_cv)))))

    
vb_fpr, vb_tpr, _ = roc_curve(y, p_y_vb)       
vb_ard_fpr, vb_ard_tpr, _ = roc_curve(y, p_y_vb_ard)   
 
 
svc_auc = roc_auc_score(y, svc_probs)
vb_auc = roc_auc_score(y, p_y_vb)
vb_ard_auc = roc_auc_score(y, p_y_vb_ard)
# summarize scores
print('SVM-linear: ROC AUC=%.3f' % (svc_auc))
print('VB: ROC AUC=%.3f' % (vb_auc))
print('VB-ARD: ROC AUC=%.3f' % (vb_ard_auc))
# calculate roc curves
svc_fpr, svc_tpr, _ = roc_curve(y, svc_probs)

# plot the roc curve for the model
plt.figure(5)
plt.plot(svc_fpr, svc_tpr, marker='.', label='SVM-linear')
plt.plot( vb_fpr, vb_tpr, marker='.', label='VB')
plt.plot( vb_ard_fpr, vb_ard_tpr, marker='.', label='VB-ARD')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

data = confusion_matrix(y, y_vb)

df_cm = pd.DataFrame(data, columns=np.unique(y), index = np.unique(y))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size