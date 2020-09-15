# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:22:22 2020

@author: M.Havlicek
"""
import numpy as np
from scipy.special import gammaln  # log gamma function
#import warnings

class vbm(object):
    '''
    Superclass for Variational Bayes regression and logistic regression 
    '''
    def __init__(self,X,y,ard,a,b,max_iter,conv_th,verbose):
        
        self.max_iter            = max_iter
        # number of features & dimensionality
        self.X                   = X
        self.y                   = y
        self.N, self.D           = self.X.shape
        self.a                   = a
        self.b                   = b
        self.ard                 = ard
        self.lower_bound         = [np.NINF]
        self.lower_bound_pred    = [np.NINF]
        self.conv_th             = conv_th
        self.verbose             = verbose
        
    def _check_convergence(self):
        '''
        Checks convergence of lower bound
        
        Returns:
        --------
        : bool
          If True algorithm converged, if False did not.
 
        '''
        assert len(self.lower_bound) >=2, 'need to have at least 2 estimates of lower bound'
        if self.lower_bound[-1] - self.lower_bound[-2] < self.conv_th:
            return True
        return False 
    
    def _check_convergence_pred(self):
        '''
        Checks convergence of lower bound (for prediction of Logit)
        
        Returns:
        --------
        : bool
          If True algorithm converged, if False did not.
 
        '''
        assert len(self.lower_bound_pred) >=2, 'need to have at least 2 estimates of lower bound'
        if self.lower_bound_pred[-1] - self.lower_bound_pred[-2] < self.conv_th:
            return True
        return False 

# ============================================================================
class vbmr(vbm):
    '''
    Variational Bayesian optimization routine for linear regression 
    of the form y = X w + e, 

    Input variables:
    y         N-by-1 data vector
    X         N-by-D design matrix 
    Optional
    ard       flag for Automatic relevance deteremination (ARD)
              ard = 1, optimization with ARD (default)
              ard = 0, optimization without ARD 
    a0,b0     noninformative inverse-Gamma priors
    c0,d0     noninformative inverse-Gamma priors

    Output variable:
    w         vector of estimated posterior weights  
    C         Estimated posterior covariance matrix
    F         Free energy - lower bound on log-evidence

    This code closely follows derivation and implementation by Jan Drugowitsch (2014)
    Variational Bayesian inference for linear and logistic regression
 
    The generative model assumes

    p(y | x, w, tau) = N(y | w'x, tau^-1),

    with x and y being the rows of the given X and y. w and tau are assigned
    the conjugate normal inverse-gamma prior
    p(w, tau | alpha) = N(w | 0, (tau alpha)^-1 I) Gam(tau | a0, b0),
    with the hyper-prior
    p(alpha) = p(alpha | c0, d0).
    The returned posterior parameters (computed by variational Bayesian
    inference) determine a posterior of the form
    
    N(w1 | w, tau^-1 V) Gam(tau | an, bn).
    '''
    def __init__(self, X, y, ard = True, a = 1e-5, b = 1e-4, c = 1e-6, d = 1e-5,
                     max_iter = 500,
                     conv_th  = 1e-4,
                     verbose = False):

        # call to constructor of superclass
        super(vbmr,self).__init__(X,y,ard,a,b,max_iter,conv_th,verbose)
        
        # parameters of Gamma distribution for precision of likelihood
        self.c   = c
        self.d   = d      
        
    def fit(self):
        '''
        Fits variational Bayesian regression
        '''        
        # re-process data
        N, D  = self.N, self.D 
        X     = self.X
        y     = self.y
        XtX   = (X.T)@(X)
        Xty   = (X.T).dot(y)
    
        a     = self.a + N/2 
     
        if self.ard:
            d = np.ones((D,1))*self.d
            p = D
        else:
            p = 1
            d = self.d
        
        c     = self.c
        
        # iterate to find hyperparameters
        for iter in range(self.max_iter):
     
            # covariance and weights of linear model
            if self.ard:
                invC  = np.diag(c/d) + XtX
            else:
                invC  = (c/d)*np.identity(D) + XtX
    
            # posterior estimates
            C   =  np.linalg.inv(invC)
            w   =  C.dot(Xty)
            
            # parameters of noise model
            eet = np.sum((X.dot(w )- y)**2)
    
            if self.ard:
                b   = self.b + 0.5*(eet + np.sum((w**2)*c/d))
            else:
                b   = self.b + 0.5*(eet + c/d*(w.T.dot(w)))
    
            # hyperparameters of covariance prior
            if self.ard:
                c   = self.d + 1/2
                d   = self.d + 0.5*(a/b*w**2 + np.diag(C))
            else:
                c   = self.c + D/2
                d   = self.c + 0.5*(a/b*(w.T.dot(w)) + np.trace(C))

            # Calculate lower bound reusing previously calculated statistics
            self._lower_bound(X,eet,C,invC,a,b,c,d,p)
                  
            # check convergence
            converged = self._check_convergence()   
            # Free energy must grow (for linear models)
            if self.verbose:
                print('Iteration {0} is completed, lower bound equals {1}'.format(iter,self.lower_bound[-1]))
                      
            if (converged or iter == (self.max_iter-1)):
                if self.verbose:
                    print('Converged')    
                # add const term to the lower bound
                self._lower_bound_const(p)    
                # save parameters of Gamma distribution
                self.a, self.b, self.c, self.d  = a, b, c, d
                # save parametres of posterior distribution 
                self.w, self.C, self.invC = w, C, invC       
                break

                    
    def _lower_bound(self,X,eet,C,invC,a,b,c,d,p): 
        
        F       = -0.5*(a/b*eet + np.sum(np.sum(X*(X.dot(C))))) - 0.5*logdet(invC) - \
                      self.b * a/b + gammaln(a) - a*np.log(b) + \
                      a + p*gammaln(c) - c*np.sum(np.log(d)) 
                    
        # lower bound        
        self.lower_bound.append(F)
    def _lower_bound_const(self,p):
        F_const = - 0.5*(self.N*np.log(2*np.pi) - self.D) - gammaln(self.a) + \
                         self.a*np.log(self.b) + p*(-gammaln(self.c) + \
                         self.c*np.log(self.d)) 
                    
        # lower bound add constants        
        self.lower_bound += F_const

    def predict(self,X):
        '''
        returns the posterior for vb_linear_fit[_ard], given the inputs x being
        the rows of X.
        
        The function expects the arguments
        - X: K x D matrix of K input samples, one per row
        - w: D-element posterior weight mean
        - V: D x D posterior weight covariance matrix
        - an, bn: scalar posterior parameter of noise precision
        w, V, an and bn are the fitted model parameters returned by
        vb_linear_fit[_ard].
        
        It returns
        - y_hat: K-element predicted output mean vector
        - y_hat_sd- K-element predicted output standard deviation vector
        
        The arguments are the ones returned by bayes_linear_fit(_ard), specifying
        the parameter posterior
        
        N(w1 | w, tau^-1 V) Gam(tau | an, bn).
        
        The predictive posteriors are of the form
        
        St(y | mu, lambda, nu),
        
        which is a Student's t distribution with mean mu, precision lambda, and
        nu degrees of freedom. All of mu and lambda a vectors, one per input x.
        nu is a scalar as it is the same for all x.
        '''
        y_hat = X.dot(self.w)

        return y_hat

    def predict_dist(self,X):
        y_hat = X.dot(self.w)
        lam   = (self.a/self.b)/(1+np.sum(X*(X.dot(self.C)),axis=1))
        nu    = 2*self.a
        y_hat_sd = np.sqrt(nu/(lam*(nu - 2)))
        return y_hat, y_hat_sd
# =============================================================================
class vbml(vbm):
    
    ''' Variational Bayesian logistic regression
    
    returns parpameters of a fitted logit model
    
    p(y = 1 | x, w) = 1 / (1 + exp(- w' * x))
    
    with a shrinkage prior on w.
    
    The function expects the arguments
    - X: N x D matrix of training input samples, one per row
    - y: N-element column vector of corresponding output {-1, 1} samples
    - a0, b0 (optional): scalar shrinkage prior parameters
    If not given, the prior/hyper-prior parameters default to a0 = 1e-2,
    b0 = 1e-4, resulting in an weak shrinkage prior.
    
    It returns
    - w: posterior weight D-element mean vector
    - C: posterior weight D x D covariance matrix
    - invC, logdetC: inverse of C, and its log-determinant
    - E_a: scalar mean E(a) of shrinkage posterior
    - F: variational bound, lower-bounding the log-model evidence p(y | X)
    
    The underlying generative model assumes a weight vector prior
    
    p(w | a) = p(w | 0, a^-1 I),
    
    and hyperprior
    
    p(a) = Gam(a | a0, b0).
    
    The function returns the parameters of the posterior:
    
    p(w1 | X, y) = N(w1 | w, C)

    '''
    def __init__(self, X, y, ard = True, a = 1e-2, b = 1e-4, 
                     max_iter = 500,
                     conv_th  = 1e-4,
                     verbose = False):

        # call to constructor of superclass
        super(vbml,self).__init__(X,y,ard,a,b,max_iter,conv_th,verbose)

    def fit(self):
        '''
        Fits variational Bayesian logistic regression
        '''
        # pre-compute some constants
        N, D  = self.N, self.D
        X     = self.X
        y     = self.y
        
        t_w = 0.5 *np.sum(X*y,axis=0).T
    
        # start first iteration here, with xi = 0 -> lam_xi = 1/8
        lam_xi = np.ones((N,1))/8
    
        if self.ard:  # using automatic relevence determination
            a = self.a + 0.5
            E_a  = np.ones((D,1))*self.a/self.a
            gammaln_a_a = D*(gammaln(a) + a)
            invC = np.diag(E_a) + 2*X.T@(X*lam_xi)
            C    = np.linalg.inv(invC)
            w    = C@t_w
            b    = self.b + 0.5*(w**2 + np.diag(C))
        else:
            a    = self.a + 0.5*D
            E_a  = self.a/self.b
            gammaln_a_a = gammaln(a) + a
            invC = E_a*np.identity(D) + 2*X.T@(X*lam_xi)
            C    = np.linalg.inv(invC)
            w    = C@t_w
            b    = self.b + 0.5*(w.T@w + np.trace(C))
        
        # update xi, b, (C, w) iteratively
        for iter in range(self.max_iter):
            # update xi by EM-algorithm
            xi     = np.sqrt(np.sum(X*(X@(C + np.outer(w,w))), axis=1,keepdims=True))
            lam_xi = lam(xi)
    
            # update posterior parameters of a based on xi
            # recompute posterior parameters of w
            if self.ard:
                b    = self.b + 0.5 * (w**2 + np.diag(C))
                E_a  = a/b
                invC = np.diag(E_a) + 2*X.T@(X*lam_xi)
            else:
                b    = self.b + 0.5 * (w.T@w + np.trace(C))
                E_a  = a/b
                invC = E_a*np.identity(D) + 2*X.T@(X*lam_xi)
            
            C       = np.linalg.inv(invC)
            logdetC = - logdet(invC)
        
            w = C@t_w
            
            # Calculate lower bound reusing previously calculated statistics
            self._lower_bound(w,xi,lam_xi,C,invC,logdetC,E_a,a,b,gammaln_a_a)
                  
            # check convergence
            converged = self._check_convergence()   
            # Free energy must grow (for linear models)
            if self.verbose:
                print('Iteration {0} is completed, lower bound equals {1}'.format(iter,self.lower_bound[-1]))
                      
            if (converged or iter == (self.max_iter-1)):
                if self.verbose:
                    print('Converged')    
                 
                # add const term to the lower bound
                self._lower_bound_const()    
                # save parameters of Gamma distribution
                self.a, self.b  = a, b
                # save parametres of posterior distribution 
                self.w, self.C, self.invC = w, C, invC   
                break
    
    def _lower_bound(self,w,xi,lam_xi,C,invC,logdetC,E_a,a,b,gammaln_a_a): 
     
        F = - np.sum(np.log(1 + np.exp(- xi))) + np.sum(lam_xi*xi**2) + \
                  0.5 * (w.T@invC@w + logdetC - np.sum(xi)) - \
                  np.sum(E_a*self.b) - np.sum(a * np.log(b)) + gammaln_a_a
                  
        # lower bound (without constant terms)  
        self.lower_bound.append(F)
                   
    def _lower_bound_const(self):
        if self.ard:
            F_const = - self.D * (gammaln(self.a) - self.a * np.log(self.b));
        else:
            F_const = - gammaln(self.a) + self.a * np.log(self.b);

        # lower bound including constant terms        
        self.lower_bound += F_const        
        


    def predict(self,X):
        ''' 
        returns a vector containing p(y=1 | x, X, Y) for x = each row in the
        given X, for a fitted Bayesian logit model.
    
        The function expects the arguments
        - X: K x D matrix of K input samples, one per row
        - w: D-element posterior weight mean
        - C: D x D posterior weight covariance matrix
        - invV: inverse of C
        w, C and invC are the fitted model parameters returned by vb_logit_fit[_*].
    
        It returns
        - out: K-element vector, with p(y=1 | x, X, Y) as each element.
    
        The function assumes model parameters corresponding to the data
        likelihood
    
        p(y = 1 | x, w1) = 1 / (1 + exp(- w1' * x)),
    
        with w, C, invV specifying the posterior parameters N(w1 | w,' C)'
        '''
    
        
        N       = self.N
        if self.N != X.shape[0]:
            N = X.shape[0]
        else:
            N = self.N
            
        w        = self.w
        C        = self.C
        invC     = self.invC
        
    
        # precompute some constants
        w_t    = np.ones((N, 1))*(w.T@invC) + 0.5 * X # w_t = C^-1 w + x / 2 as rows
        Cx     = X @ C                                 # W x as rows
        CxxCwt = Cx*(np.sum(w_t * Cx, axis=1)[:,np.newaxis]) # C x x^T C^T w_t as rows
        Cwt    = w_t @ C                               # C w_t as rows
        xCx    = np.sum(Cx * X,axis=1)                 # x^T C x as rows
        xCx2   = xCx**2                              # x^T C x x^T C x as rows
    
        # start first iteration with xi = 0, lam_xi = 1/8
        xi     = np.zeros((N, 1))
        lam_xi = np.ones((N,1))*1e-16
        a_xi   = 1/(4 + xCx)
        w_xi   = Cwt - (a_xi[:,np.newaxis] * CxxCwt)
        logdetC_xi = - np.log(1 + xCx/4)
        wCw_xi = np.sum(w_xi * (w_xi @ invC), axis=1) + (np.sum(w_xi * X, axis=1)**2)/4
        F      = 0.5 * (np.sum(logdetC_xi) + np.sum(wCw_xi)) - N*np.log(2)
        self.lower_bound_pred.append(F)
    
        # iterate to from xi's that maximise variational bound
        for iter in range(self.max_iter):
            #update xi by EM algorithm
            xi     = np.sqrt(xCx - a_xi * xCx2 + np.sum(w_xi * X, axis=1)**2)
            lam_xi = lam(xi)
    
            #Sherman Morrison formula and Matrix determinant lemma
            a_xi       = 2*lam_xi/(1 + 2 * lam_xi * xCx)
            w_xi       = Cwt - (a_xi[:,np.newaxis] * CxxCwt)
            logdetC_xi = - np.log(1 + 2 * lam_xi * xCx)
    
            #variational bound, omitting constant terms
            wCw_xi = np.sum(w_xi * (w_xi @ invC), axis=1) + \
                     2 * lam_xi * (np.sum(w_xi * X, axis=1)**2)
 
            # Calculate lower bound reusing previously calculated statistics
            self._lower_bound_pred(logdetC_xi,wCw_xi,xi,lam_xi)
                  
            # check convergence
            converged = self._check_convergence_pred()   
            # Free energy must grow (for linear models)
            if self.verbose:
                print('Iteration {0} is completed, lower bound equals {1}'.format(iter,self.lower_bound_pred[-1]))
                      
            if (converged or iter == (self.max_iter-1)):
                if self.verbose:
                    print('Converged')    
                 
                break    

        # posterior probability from optimal xi's
        out = 1 / (1 + np.exp(-xi)) / np.sqrt(1 + 2 * lam_xi * xCx) * \
                   np.exp(0.5 * (-xi - w.T @ invC @ w + wCw_xi) + lam_xi * xi**2)
        return out

    def _lower_bound_pred(self,logdetC_xi,wCw_xi,xi,lam_xi): 
     
        F = np.sum(0.5 * (logdetC_xi + wCw_xi - xi) - \
                        np.log(1 + np.exp(-xi)) + lam_xi * xi**2)
        # lower bound (without constant terms)  
        self.lower_bound_pred.append(F)
# ======================== Helper Functions =====================================
    
# def sigmoid(theta):
#     '''
#     Sigmoid function
#     '''
#     return 1./( 1 + np.exp(-theta))

# def lam(xi):
#     '''
#     Helper function for local variational approximation of sigmoid function
#     '''
#     return 0.5 / eps * ( sigmoid(xi) - 0.5)


def logdet(A):
    #computes the log(det(A)) of a positive definite A.'''
    out = 2 *np.sum(np.log(np.diag(np.linalg.cholesky(A))))
    return out
 
def lam(xi):
    out = np.tanh(xi/2)/(4*xi)
    out = np.nan_to_num(out)
    return out  