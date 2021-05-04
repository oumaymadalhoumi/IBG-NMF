"""
Online implementation of Variational Bayesian inference for non-negative matrix factorisation.
We expect the following arguments:
- R, the matrix.
- M, the mask matrix indicating observed values (1) and unobserved ones (0).
- K, the number of latent factors.
- hyperparameters = { 'alpha', 'nu', 'beta', 'mu', 'rho', 'zeta' },
    

Usage of class:
    BNMF = ib_bnmf(R,M,K,hyperparameters)
    BNMF.initisalise()      
    BNMF.run(iterations)
Or:
    BNMF = ib_bnmf(R,M,K,hyperparameters)
    BNMF.train(iterations)

We can test the performance of our model on a test dataset, specifying our test set with a mask M. 
    performance = BNMF.predict(M_pred)
This gives a dictionary of performances,
    performance = { 'MSE', 'R^2', 'Rp' }
    
The performances of all iterations are stored in BNMF.all_performances, which 
is a dictionary from 'MSE', 'R^2', or 'Rp' to a list of performances.
    
Finally, we can return the goodness of fit of the data using the quality(metric) function:
- metric = 'loglikelihood' -> return p(D|theta)
         = 'BIC'        -> return Bayesian Information Criterion
         = 'AIC'        -> return Afaike Information Criterion
         = 'MSE'        -> return Mean Square Error
         = 'ELBO'       -> return Evidence Lower Bound
"""

import sys 
sys.path.append('/home/oumayma/Desktop/MSc/research/git/')

from IBG_BNMF.code.distributions.gamma_vector import gamma_vector_expectation
import math, time
from scipy.special import digamma 
import numpy as np
from tqdm import tqdm
import random 


ALL_METRICS = ['MSE','R^2','Rp']
ALL_QUALITY = ['loglikelihood','BIC','AIC','MSE','ELBO']
OPTIONS_INIT_UV = ['random', 'exp']

class ib_bnmf_online:
    def __init__(self,R,M,K,hyperparameters , online_param  ):
        ''' Set up the class and do some checks on the values passed. '''
        self.R = np.array(R,dtype=float)
        self.M = np.array(M,dtype=float)
        self.K = K
        assert len(self.R.shape) == 2, "Input matrix R is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.R.shape)
        assert self.R.shape == self.M.shape, "Input matrix R is not of the same size as " \
            "the indicator matrix M: %s and %s respectively." % (self.R.shape,self.M.shape)
            
        (self.I,self.J) = self.R.shape
        self.check_empty_rows_columns()      
        
        #Initialize Parameters 
        self.alphaA_0 , self.muA_0 = hyperparameters['alphaA'], hyperparameters['muA']
        self.betaB_0, self.nuB_0 = hyperparameters['betaB'], hyperparameters['nuB']
        self.zetaH_0, self.rhoH_0 = hyperparameters['zetaH'], hyperparameters['rhoH']
    
        
        self.alphaA, self.muA = hyperparameters['alphaA'], hyperparameters['muA']
        self.betaB, self.nuB = hyperparameters['betaB'], hyperparameters['nuB']
        self.zetaH, self.rhoH = hyperparameters['zetaH'], hyperparameters['rhoH']
        

        self.len_Is =     online_param['len_Is']
        self.len_Js =     online_param['len_Js']
        self.rate  = online_param['rate']
        self.eps = self.rate

        self.iii = 0

    def check_empty_rows_columns(self):
        ''' Raise an exception if an entire row or column is empty. '''
        sums_columns = self.M.sum(axis=0)
        sums_rows = self.M.sum(axis=1)
                    
        # Assert none of the rows or columns are entirely unknown values
        for i,c in enumerate(sums_rows):
            assert c >= 0, "Fully unobserved row in R, row %s." % i
        for j,c in enumerate(sums_columns):
            assert c >= 0, "Fully unobserved column in R, column %s." % j

    def train(self,init_UV,iterations):
        ''' Initialise and run the algorithm. '''
        self.initialise(init_UV=init_UV)
        self.run_full_batch(iterations)

    def initialise(self,init_UV='exp'):
        ''' Initialise U, V, tau, and lambda (if ARD). '''
        assert init_UV in OPTIONS_INIT_UV, "Unknown initialisation option: %s. Should be in %s." % (init_UV, OPTIONS_INIT_UV)
        
        # Update zero values in the input Matrix to the minimum value divided by 10 to avoid log(zero) 
        self.R = np.where(self.R > 0 , self.R , np.min(self.R[np.nonzero(self.R)])/10)
        # Check if R has zero values 
        assert np.min(self.R) > 0, "Input matrix R has zero or negative values, " 

        # Compute expectations and variances U, V
        self.exp_A = np.zeros((self.I,self.K))
        self.exp_B = np.zeros((self.I,self.K))
        self.exp_H = np.zeros((self.K,self.J))
        
        self.update_exp_A()
        self.update_exp_B()
        self.update_exp_H()
        
        self.update_digamma_AH()
        self.update_digamma_BH()
        self.update_digamma_ABH()

    def run_full_batch(self,iterations):
        ''' Run the VI implementation. '''
        self.all_performances = {} # for plotting convergence of metrics
        for metric in ALL_METRICS:
            self.all_performances[metric] = []

        start_time = time.time()

        for it in tqdm(range(iterations)): 

            # Update A
            self.update_A()
            self.update_exp_A()    
            self.update_digamma_AH()
            self.update_digamma_ABH()

            # Update B
            self.update_B()
            self.update_exp_B()
            self.update_digamma_BH()
            self.update_digamma_ABH()
            
            # Update H
            self.update_H()
            self.update_exp_H()
            self.update_digamma_AH()
            self.update_digamma_BH()
            self.update_digamma_ABH()
            
            
            # Store and print performances
            perf= self.predict(self.M) 
            for metric in ALL_METRICS:
                self.all_performances[metric].append(perf[metric])
            
            #print ("Iteration %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,perf['MSE'],perf['R^2'],perf['Rp']))
            
        print("--- This run took %s seconds ---" % (time.time() - start_time))


    def run_online(self, l1,l2, rate , R = None, hyperparameters = None ) :
        
        self.rate = rate
        
        if R is not None: # If R was updated 
            self.R = R 

            M =np.copy(self.M) 

            if R.shape[0] > self.I : 
                self.M  = np.append(M, np.zeros((R.shape[0] -self.I , self.J)), axis = 0)
    
                alphaA_0 , muA_0 = hyperparameters['alphaA'], hyperparameters['muA']
                betaB_0, nuB_0 = hyperparameters['betaB'], hyperparameters['nuB']

            
                self.alphaA_0 = np.append(self.alphaA_0 , alphaA_0 , axis = 0)
                self.muA_0 = np.append(self.muA_0 , muA_0 , axis = 0)
                self.betaB_0 = np.append(self.betaB_0 , betaB_0 , axis = 0)
                self.nuB_0 = np.append(self.nuB_0 , nuB_0 , axis = 0)

                self.alphaA = np.append(self.alphaA , alphaA_0 , axis = 0)
                self.muA = np.append(self.muA , muA_0 , axis = 0)
                self.betaB = np.append(self.betaB ,betaB_0 , axis = 0)
                self.nuB = np.append(self.nuB, nuB_0, axis = 0)

            if R.shape[1] > self.J : 
                M =np.copy(self.M) 
                self.M  = np.append(M, np.zeros((M.shape[0] , R.shape[0] -self.J)), axis = 1)
    
                zetaH_0, rhoH_0 = hyperparameters['zetaH'], hyperparameters['rhoH']

                self.zetaH_0 = np.append(self.zetaH_0 , zetaH_0 ,axis = 1)
                self.rhoH_0 = np.append(self.rhoH_0 , rhoH_0 ,axis = 1)

                self.zetaH = np.append(self.zetaH , zetaH_0 ,axis = 1)
                self.rhoH = np.append(self.rhoH , rhoH_0 ,axis = 1)
                
            self.I , self.J = self.R.shape
            self.update_exp_A()
            self.update_exp_B()
            self.update_exp_H()
            
            self.update_digamma_AH()
            self.update_digamma_BH()
            self.update_digamma_ABH()
            
            del M

        self.M[l1,l2] = 1

        self.all_performances = {} # for plotting convergence of metrics
        for metric in ALL_METRICS:
            self.all_performances[metric] = []

        self.update_A_online(l2)
        self.update_exp_A()    
        self.update_digamma_AH()
        self.update_digamma_ABH()

        # Update B
        self.update_B_online(l2)
        self.update_exp_B()
        self.update_digamma_BH()
        self.update_digamma_ABH()
        
        # Update H
        self.update_H_online(l1)
        self.update_exp_H()
        self.update_digamma_AH()
        self.update_digamma_BH()
        self.update_digamma_ABH()
        
        
        # Store and print performances
        perf= self.predict(self.M) 
        for metric in ALL_METRICS:
            self.all_performances[metric].append(perf[metric])
        
        #print ("Iteration %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,perf['MSE'],perf['R^2'],perf['Rp']))
        

    def update_A(self):   
        ''' Parameter updates A. '''            
        self.alphaA = self.alphaA_0 - np.dot(self.M*(np.log(self.R)-np.log(1+self.R)), self.exp_H.transpose())
        self.muA = self.muA_0 + self.exp_A*(np.dot(self.M*(self.digamma_ABH-self.digamma_AH),self.exp_H.transpose()))


    def update_A_online(self, J_s):   
        ''' Parameter updates A. '''           
        alpha_up = np.zeros(self.exp_A.shape)
        mu_up = np.zeros(self.exp_A.shape) 
        JJ_s = len(J_s)

        for j in J_s : 
            alpha_up += np.dot(self.M[:,j:j+1]*(np.log(self.R[:,j:j+1])-np.log(1+self.R[:,j:j+1])), self.exp_H[:,j:j+1].transpose())
            
            mu_up += self.exp_A*(np.dot(self.M[:,j:j+1]*(self.digamma_ABH[:,j:j+1]-self.digamma_AH[:,j:j+1]),self.exp_H[:,j:j+1].transpose()))
    

        alphaA1 = self.alphaA_0 - (self.J / JJ_s) * alpha_up
        muA1 = self.muA_0 + (self.J / JJ_s) * mu_up        

        self.alphaA = (1-self.rate)*self.alphaA + self.rate * alphaA1
        self.muA = (1-self.rate)*self.muA + self.rate * muA1        

        #self.alphaA = self.alphaA_0 - np.dot(self.M*(np.log(self.R)-np.log(1+self.R)), self.exp_H.transpose())
        #self.muA = self.muA_0 + self.exp_A*(np.dot(self.M*(self.digamma_ABH-self.digamma_AH),self.exp_H.transpose()))
            
    def update_B(self):   
        ''' Parameter updates A. '''            
        self.betaB = self.betaB_0 + np.dot(self.M*(np.log(1+self.R)), self.exp_H.transpose())
        self.nuB = self.nuB_0 + self.exp_B*(np.dot(self.M*(self.digamma_ABH-self.digamma_BH),self.exp_H.transpose()))


    def update_B_online(self,J_s):   
        ''' Parameter updates A. '''    

        beta_up = np.zeros(self.exp_B.shape)
        nu_up = np.zeros(self.exp_B.shape)
        JJ_s = len(J_s)

        for j in J_s : 
            beta_up += np.dot(self.M[:,j:j+1]*(np.log(1+self.R[:,j:j+1])), self.exp_H[:,j:j+1].transpose())
            nu_up += self.exp_B*(np.dot(self.M[:,j:j+1]*(self.digamma_ABH[:,j:j+1]-self.digamma_BH[:,j:j+1]),self.exp_H[:,j:j+1].transpose()))

        betaB1 = self.betaB_0 + (self.J / JJ_s) * beta_up
        nuB1 = self.nuB_0 + (self.J / JJ_s) * nu_up
        
        self.betaB = (1-self.rate) * self.betaB + self.rate *betaB1
        self.nuB = (1-self.rate) * self.nuB + self.rate *nuB1


        #self.betaB = self.betaB_0 + np.dot(self.M*(np.log(1+self.R)), self.exp_H.transpose())
        #self.nuB = self.nuB_0 + self.exp_B*(np.dot(self.M*(self.digamma_ABH-self.digamma_BH),self.exp_H.transpose()))

    def update_H(self):   
        ''' Parameter updates A. '''            
        self.zetaH = self.zetaH_0 - np.dot(self.exp_A.transpose(), self.M*np.log(self.R)) + np.dot((self.exp_A+self.exp_B).transpose(),self.M*np.log(self.R+1.0))
        
        self.rhoH = self.rhoH_0 + self.exp_H*( np.dot((self.exp_A).transpose(), self.M*(self.digamma_ABH -self.digamma_AH)) + np.dot((self.exp_B).transpose(), self.M*(self.digamma_ABH -self.digamma_BH)) )   


    def update_H_online(self, I_s):   
        ''' Parameter updates A. ''' 

        rho_up = np.zeros(self.exp_H.shape)
        zeta_up = np.zeros(self.exp_H.shape)
        II_s = len(I_s)
        
        for i in I_s : 
            zeta_up += np.dot(self.exp_A[i:i+1,:].transpose(), self.M[i:i+1,:]*np.log(self.R[i:i+1,:])) - np.dot((self.exp_A[i:i+1,:]+self.exp_B[i:i+1,:]).transpose(),self.M[i:i+1,:]*np.log(self.R[i:i+1,:]+1.0))    

            rho_up +=  self.exp_H*( np.dot((self.exp_A[i:i+1,:]).transpose(),self.M[i:i+1,:]*(self.digamma_ABH[i:i+1,:] -self.digamma_AH[i:i+1,:])) + np.dot((self.exp_B[i:i+1,:]).transpose(), self.M[i:i+1,:]*(self.digamma_ABH[i:i+1,:] -self.digamma_BH[i:i+1,:] )) ) 

        zetaH1 = self.zetaH_0 - (self.I /II_s ) * zeta_up
        rhoH1 = self.rhoH_0 + (self.I /II_s ) * rho_up

        self.zetaH = (1-self.rate) * self.zetaH + self.rate *zetaH1
        self.rhoH = (1-self.rate) * self.rhoH + self.rate *rhoH1

    def update_digamma_AH(self) : 
        self.digamma_AH = digamma(np.dot(self.exp_A,self.exp_H))

    def update_digamma_BH(self) : 
        self.digamma_BH = digamma(np.dot(self.exp_B,self.exp_H))

    def update_digamma_ABH(self) : 
        self.digamma_ABH = digamma(np.dot(self.exp_A+self.exp_B,self.exp_H))


    def update_exp_A(self):
        ''' Update expectation U. '''
        self.exp_A = gamma_vector_expectation(self.muA,self.alphaA)
        #self.var_U[:,k] = TN_vector_variance(self.mu_U[:,k],self.tau_U[:,k])


    def update_exp_B(self):
        ''' Update expectation U. '''
        self.exp_B = gamma_vector_expectation(self.nuB,self.betaB)
        #self.var_U[:,k] = gamma_vector_variance(self.mu_U[:,k],self.tau_U[:,k])

    def update_exp_H(self):
        ''' Update expectation U. '''
        self.exp_H = gamma_vector_expectation(self.rhoH,self.zetaH)
        #self.var_U[:,k] = gamma_vector_variance(self.mu_U[:,k],self.tau_U[:,k])

    def normalize(self, M ) : 
            return (M / M.sum(axis=1)[:, np.newaxis])       

    def predicted_R(self):
        ''' Predict missing values in R. '''
        nominator = np.dot(self.exp_A , self.exp_H)
        denominator = np.dot(self.exp_B , self.exp_H) - np.float64(1.0)
        return np.divide ( nominator, denominator )

    def predict(self, M_pred):
        ''' Predict missing values in R. '''
        R_pred = self.predicted_R()
        MSE = self.compute_MSE(M_pred, self.R, R_pred)
        R2 = self.compute_R2(M_pred, self.R, R_pred)    
        Rp = self.compute_Rp(M_pred, self.R, R_pred)        
        return {'MSE': MSE, 'R^2': R2, 'Rp': Rp}
        
    ''' Functions for computing MSE, R^2 (coefficient of determination), Rp (Pearson correlation) '''
    def compute_MSE(self,M,R,R_pred):
        ''' Return the MSE of predictions in R_pred, expected values in R, for the entries in M. '''
        return (M * (R-R_pred)**2).sum() / float(M.sum())
        
    def compute_R2(self,M,R,R_pred):
        ''' Return the R^2 of predictions in R_pred, expected values in R, for the entries in M. '''
        mean = (M*R).sum() / float(M.sum())
        SS_total = float((M*(R-mean)**2).sum())
        SS_res = float((M*(R-R_pred)**2).sum())
        return 1. - SS_res / SS_total if SS_total != 0. else np.inf
        
    def compute_Rp(self,M,R,R_pred):
        ''' Return the Rp of predictions in R_pred, expected values in R, for the entries in M. '''
        mean_real = (M*R).sum() / float(M.sum())
        mean_pred = (M*R_pred).sum() / float(M.sum())
        covariance = (M*(R-mean_real)*(R_pred-mean_pred)).sum()
        variance_real = (M*(R-mean_real)**2).sum()
        variance_pred = (M*(R_pred-mean_pred)**2).sum()
        return covariance / float(math.sqrt(variance_real)*math.sqrt(variance_pred))