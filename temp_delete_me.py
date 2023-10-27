from operator import index
import sklearn.model_selection as skm
import sklearn.metrics as smm
from statistics import mean
import numpy as np
import numpy.random as npr
import xgboost as xgb
import dataloader_varity as data
import weighting_functions as wf


varity_data = data.Dataloader_Varity("/Users/alirezarasoulzadeh/Desktop/reimplemented_varity/test_config.json")

def core_set_finder(data: data.Dataloader_Varity)->np.array:
    '''data -> Dataloader_Varity instance 
        this function returns a numpy array that contains all of the indices of core set members'''
    qip_dict = varity_data.qip_dict
    core_sets = qip_dict['core'].keys()
    index_array = np.array([])
    for core_set in core_sets:
        temp_array = (data.data['set_name']==core_set).to_numpy().nonzero()[0] #get an array of indices corresponding to a core set
        #get first element, since we are dealing with 1D arrays and nonzero returns a tuple
        index_array = np.concatenate((index_array,temp_array))
    
    return index_array


def core_targeted_CV(indices_and_params:dict):
    '''vanilla cross validation with the loss returned being an average over all test sets
    only core set members are used for CV'''
    
    #varity_dataloader instance from param dict
    varity_data = indices_and_params["varity_data"]
    errors_list = []
    kf = skm.KFold(n_splits=5)
    parameter_dict = {"eta":indices_and_params['eta'], "gamma": indices_and_params['gamma'], "max_depth":indices_and_params['max_depth'], "min_child_weight":indices_and_params['min_child_weight'],
    "subsample":indices_and_params['subsample'], "colsample_bytree":indices_and_params['colsample_bytree'],"tree_method":'exact',"objective":"reg:logistic"}
    varity_featurelist = list(varity_data.feature_set)
    varity_r_data_unsplit_labelled = varity_data.data[varity_featurelist+['label']]
    
    weights = wf.Weight(varity_data.data,varity_data.qip_dict)
    weights.fw_core_multiply_weight_vector_maker(varity_data.data, varity_data.qip_dict,indices_and_params,False)

    #get core set indices for splitting only core instances
    indices_core_set = core_set_finder(varity_data)
    #create an array with integers ranging from 0 to num rows in training data so that we can create a train set  
    dummy_index = np.arange(varity_data.data.shape[0])
    for train, test in kf.split(indices_core_set):
        test_indices = indices_core_set[test]
        train_indices = indices_core_set[train]
        #np.in1d compares two arrays and returns a boolean with anywhere elements in arg2 appear in arg1
        train_indices = np.nonzero(~np.in1d(np.arange(varity_data.data.shape[0]),train_indices))[0]
        weights_train = weights.weights[train_indices]
        weights_test = weights.weights[test_indices]
        train = varity_r_data_unsplit_labelled.iloc[train_indices]
        test = varity_r_data_unsplit_labelled.iloc[test_indices]


        train_dataset = xgb.DMatrix(train[varity_featurelist],weight=weights_train, label=train["label"],feature_names=varity_featurelist)
        test_dataset = xgb.DMatrix(test[varity_featurelist],weight=weights_test, label=test["label"],feature_names=varity_featurelist)

        model = xgb.train(parameter_dict,train_dataset,num_boost_round=indices_and_params['num_boost_round'])

        predicted_labels = model.predict(test_dataset)
        print(predicted_labels.shape)
        prior = prior_calc(test)
        error = aubprc(test["label"],predicted_labels,prior)
        print(prior)
        errors_list.append(error)

    print(errors_list)
    return {'loss':-1*mean(errors_list), 'status':STATUS_OK}