import sys 
sys.path.append("..\\")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold

from models.IBG_BNMF import ib_nmf
from models.online_IBG_BNMF import ib_nmf_online

from data.CF import CF

# Read dataset 
path = '../data/movielens-dataset/ratings.dat'
sep = "::"
columns = ["user_id", "item_id",  "rating", "timestamp"] 	
df_raw = pd.read_csv(path, sep , names = columns, dtype = {'rating':np.int} ,header=None , usecols=[0, 1, 2, 3])

#Create matrices 
mvlens =  CF(df_raw)

#Prepare Hyperparameters   
I = mvlens.count_users
J = mvlens.count_movies

#define model params
list_k = [50,100,200,800]
iterations = 100

#define variables to save evaluation metrics
MSE_test_ib = {k:[] for k in list_k}
MSE_train_ib = {k:[] for k in list_k}

kf = KFold(10, shuffle = True ) # 10 fold cross validation 
fold = 0
for train_index, test_index in kf.split(mvlens.data) : 
    fold += 1 
    R, M_mask, M_test = mvlens.build_user_item_matrix(train_index, test_index, full_R = True) 
    R = R.toarray()
    M_mask = M_mask.toarray()
    M_test = M_test.toarray()
     
    for K in list_k: 
        mult_alpha_beta = 1*10/K
        mult_rho = 1
        
        #Prepare Hyperparameters Matrices  
        mu_0 = np.random.rand(I,K)*mult_alpha_beta# sh ape parameter
        alpha_0 = np.full((I,K),1) # inverse scale parameter
        
        nu_0 = np.random.rand(I,K)*mult_alpha_beta # shape parameter
        beta_0 = np.full((I,K),1) # inverse scale parameter
        
        rho_0 = np.random.rand(K,J ) * mult_rho # shape parameter 
        zeta_0 = np.full(( K,J),1 ) # inverse scale parameter 
        
        hyperparameters = {}
        hyperparameters['alphaA'], hyperparameters['muA'] = alpha_0, mu_0
        hyperparameters['betaB'], hyperparameters['nuB'] = beta_0, nu_0
        hyperparameters['zetaH'], hyperparameters['rhoH'] = zeta_0, rho_0
        del  mu_0, nu_0, beta_0, alpha_0, zeta_0, rho_0 
    
        ###############################################################
        ############### INVERTED BETA IMPLEMENTATION ##################
        ###############################################################
        ib = ib_nmf( R=R, M=M_mask, K=K , hyperparameters = hyperparameters)
        ib.train(init_UV="exp", iterations = iterations)
        
        MSE_train_ib[K].append(ib.predict(M_mask)['MSE'])
        MSE_test_ib[K].append(ib.predict(M_test)['MSE'])

        f = open("..\\results\\1M\\MSE_train_ib1.txt","a")
        f.write( "%s,%s,%s"  % (fold ,K ,  MSE_train_ib[K][fold-1]) )
        f.write( '\n' )
        f.close()
