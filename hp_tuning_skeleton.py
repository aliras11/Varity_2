'''a set of functions used to try out different cross validation schemes, for checking model performance and\
    hp tuning'''

import sklearn.model_selection as skm
from statistics import mean
import numpy as np
import numpy.random as npr
import xgboost as xgb



def create_data(miu1=0,miu2=1,std1=1,std2=1,size=100,prior=0.5):
    '''create a binary classification problem with one feature that is drawn from two normal distributions'''
    norm1 = npr.normal(miu1,std1,round(size*prior)) #feature for 0 class
    norm2 = npr.normal(miu2,std2,size-round(size*(prior))) #feature for 1 class
    feature = np.r_[norm1,norm2]
    labels = np.r_[np.repeat(0,round(size*prior)),np.repeat(1,size-round(size*(prior)))]

    return feature,labels

#Create the different CV options

def cv_mean_testset(eta,gamma,max_depth,min_child_weight,subsample,colsample_bytree,tree_method,objective,n_splits,data,weight_function):
    '''vanilla cross validation with the loss returned being an average over all test sets
        all of data gets to be a test set once'''
    loss_list = []
    parameter_dict = {'eta':eta,'gamma':gamma,'max_depth':max_depth,'min_child_weight':min_child_weight,'subsample':subsample,
    'colsample_bytree':colsample_bytree,"tree_method":tree_method,"objective":objective}
    kf = skm.KFold(n_splits=n_splits)
    for train, test in kf.split(data):
        


    
def nested_cv_():
    '''nested cross validation - find the best set of hyperparameters on outer train set then validate
    on outer loop test set. This adds a multiplicative factor on the number of times the model is fit. 
    '''

def 