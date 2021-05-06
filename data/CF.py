
import sys 
sys.path.append('..\\')
sys.path.append('.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import scipy.sparse as sparse

from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle

split_methods = ["random", "k_fold"]

class CF :
    
    def __init__(self, data,hyperparam={} ): 
        self.prepare_raw_data( data)
        

    def prepare_raw_data (self,df_raw): 
        # Create user- & movie-id mapping
        user_id_mapping = {id:i for i, id in enumerate(df_raw['user_id'].unique())}
        movie_id_mapping = {id:i for i, id in enumerate(df_raw['item_id'].unique())}

        df_raw['user_id'] = df_raw['user_id'].map(user_id_mapping)
        df_raw['item_id'] = df_raw['item_id'].map(movie_id_mapping)
        df_raw['mask'] = [1]*len(df_raw) 
        
        self.user_id_mapping = user_id_mapping 
        self.movie_id_mapping = movie_id_mapping 

        self.count_users = len(self.user_id_mapping)
        self.count_movies = len(self.movie_id_mapping)

        self.data = shuffle(df_raw)        

        
    def split_data (self, method,test_size = 0.20, k=5 ) :    
        assert method in split_methods , "'method' should be 'random' or 'k_fold', but is %s instead." %method
        # make sure all users and all movies appear in the train set 
        repeat = True
        while repeat :
            split = []
            if method == "random" : 
                df_train, df_test = train_test_split(self.data,  test_size=test_size, random_state=42)
                split.append([df_train.index, df_test.index ])
                train_users = len(df_train['user_id'].unique() )
                train_movies = len(df_train['item_id'].unique() )
                if (train_users == self.count_users) and (train_movies == self.count_movies) :
                    repeat = False
                else : 
                    print ( ' retrying ' )
                    repeat = False
            if method == "k_fold" : 
                kf = KFold(n_splits=k,shuffle = True,random_state=42)
            # Make sure that train set has all users and movies 
                repeat = False 
                for df_train, df_test in kf.split(self.data) :
                    split.append([df_train, df_test ])
                    train_users = len(self.data.loc[df_train]['user_id'].unique() )
                    train_movies = len(self.data.loc[df_train]['item_id'].unique() )
                    if (train_users < self.count_users) and (train_movies < self.count_movies) :
                        print ( ' retrying ' )
                        repeat = True                 
        return split 
    
    def create_matrices(self, train_index , test_index ): 

        R = self.data.pivot_table(index='user_id', columns='item_id', values='rating',fill_value = 0,dropna = False).values
        print('Shape User-Movie-Matrix:\t{}'.format(R.shape))
        
        # M_mask will be used to train the model, it takes 1 for values from the train data, else 0
        df_train = pd.DataFrame(np.float64(0.0), index=self.data.index, columns=self.data.columns)
        df_train[["user_id", "item_id" ]]=self.data[["user_id", "item_id", ]]
        df_train.loc[train_index] = self.data.loc[train_index]
        M_mask = df_train.pivot_table(index='user_id', columns='item_id', values='mask',fill_value = 0,dropna = False).values
        del df_train 
        
        # M_test will be used to evaluate the algorithm X_test = X_pred * M_test, where X_pred is the fully predicted Matrix, X_test is the prediction for the testset
        df_test = pd.DataFrame(np.float64(0.0), index=self.data.index, columns=self.data.columns)
        df_test[["user_id", "item_id"]]=self.data[["user_id", "item_id", ]]
        df_test.loc[test_index] = self.data.loc[test_index]
        M_test = df_test.pivot_table(index='user_id', columns='item_id', values='mask',fill_value = 0,dropna = False).values
        del df_test 
        
        return R, M_mask, M_test  
                 
    
    def build_user_item_matrix(self, train_index , test_index, full_R = True):
        shape =(self.count_users ,self.count_movies )
        if full_R : 
            data = self.data['rating']
            row_ind = self.data['user_id']
            col_ind = self.data['item_id']
            
            
            test = self.data.loc[test_index]
            train = self.data.loc[train_index]
            R = sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape)
            M_mask = sparse.csr_matrix((train['mask'], (train['user_id'], train['item_id'])), shape=shape)
            M_test = sparse.csr_matrix((test['mask'], (test['user_id'], test['item_id'])), shape=shape)
            return R, M_mask, M_test

        else : 
            
            test = self.data.loc[test_index]
            train = self.data.loc[train_index]
            R_train = sparse.csr_matrix((train['rating'], (train['user_id'], train['item_id'])), shape=shape)
            R_test = sparse.csr_matrix((test['rating'], (test['user_id'], test['item_id'])), shape=shape)
            return R_train, R_test
    

    def normalize_data(self ,R, delta = 0.5 ):
        R_norm  = ( R -1 + delta )/ (4+2*delta)
        R_norm[R_norm < 0] = 0
        return R_norm
            
    
    