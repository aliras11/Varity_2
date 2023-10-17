'''a set of functions used to try out different hp tuning schemes, for checking model performance and\
    hp tuning'''

import sklearn.model_selection as skm
from statistics import mean
import numpy as np
import numpy.random as npr
import xgboost as xgb
import dataloader_varity 
import weighting_functions as wf



def cv_mean_testset(varity_data: dataloader_varity.Dataloader_Varity, indices_and_params:dict):
    '''vanilla cross validation with the loss returned being an average over all test sets
        all of data gets to be a test set once'''
    errors_list = []
    kf = skm.KFold(n_splits=10)
    parameter_dict = {"eta":indices_and_params['eta'], "gamma": indices_and_params['gamma'], "max_depth":indices_and_params['max_depth'], "min_child_weight":indices_and_params['min_child_weight'],
    "subsample":indices_and_params['subsample'], "colsample_bytree":indices_and_params['colsample_bytree'],"tree_method":'gpu_hist',"objective":"reg:logistic"}
    varity_r_data_unsplit_labelled = varity_data.data[varity_data.feature_set+['label']]
    #varity_r_data_unsplit_features_only = varity_r_data_unsplit_labelled[list_features].iloc[]
    #labels = varity_r_data_unsplit_labelled["label"]

    weights = wf.Weight(varity_data.data,varity_data.qip_dict)
    weights.fw_core_multiply_weight_vector_maker(varity_data.data, varity_data.qip_dict,indices_and_params,True)
    for train, test in kf.split(varity_r_data_unsplit_labelled):
        print(type(train))
        weights_train = weights[train]
        weights_test = weights[test]
        train = varity_r_data_unsplit_labelled.iloc[train.tolist()]
        test = varity_r_data_unsplit_labelled.iloc[test.tolist()]
        print(train.shape)

        train_dataset = xgb.DMatrix(train[list_features],weight=weights_train, label=train["label"],feature_names=list_features)
        test_dataset = xgb.DMatrix(test[list_features],weight=weights_test, label=test["label"],feature_names=list_features)

        model = xgb.train(parameter_dict,train_dataset,num_boost_round=indices_and_params['num_boost_round'])

        predicted_labels = model.predict(test_dataset)
        print(predicted_labels.shape)
        prior = prior_calc(test)
        error = aubprc(test["label"],predicted_labels,prior)
        print(prior)
        errors_list.append(error)

    print(errors_list)
    return {'loss':-1*mean(errors_list), 'status':STATUS_OK}

def nested_cv_():
    '''nested cross validation - find the best set of hyperparameters on outer train set then validate
    on outer loop test set. This adds a multiplicative factor on the number of times the model is fit. 
    '''
    pass