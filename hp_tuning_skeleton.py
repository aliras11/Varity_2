'''a set of functions used to try out different hp tuning schemes, for checking model performance and\
    hp tuning'''

import sklearn.model_selection as skm
import sklearn.metrics as smm
from statistics import mean
import numpy as np
import numpy.random as npr
import xgboost as xgb
import dataloader_varity as data
import weighting_functions as wf
import hyperopt.pyll.stochastic
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

#takes in pandas dataframe because it will have been made accessible via the appropriate indices
def prior_calc(data_frame):

    ''' calculates the ratio of positive to negative examples, p(Y=1)
    data_frame -> pandas data frame, must have a label column with 0 or 1 values
    '''
    freq_dict = data_frame["label"].value_counts().to_dict()
    if 1 in freq_dict.keys() and 0 in freq_dict.keys():
        negative = freq_dict[0]
        positive = freq_dict[1]
        prior = positive/(positive+negative) #calculate prior probability of pathogeneticity 

    elif 1 in freq_dict.keys() and 0 not in freq_dict.keys(): #if there are only pathogenic/positive labels 
        prior = 1 #prior if there are no positive cases

    else: #if there are no pathogenic/positive labels 
        prior = 0
    
    return prior

#fold specific priors needed not just a general one
def aubprc(y_true,predictions,prior):

  ''' implements the area under the balanced precision recall curve as per the varity paper
      y_true -> array-like of true labels
      predictions -> array like of predicted labels
      prior -> float calculated per fold of test set, ratio of positive to negative labels p(Y=1)
  '''
  auprc_average_precision = smm.average_precision_score(y_true, predictions)
  aubprc = (auprc_average_precision * (1-prior))/((auprc_average_precision * (1-prior))+((1-auprc_average_precision)*prior))
  return aubprc

def hp_space_builder_varity(qip_dict:dict, weight_function:str) -> dict:
  space = {
        'num_boost_round':hp.choice('num_boost_round', np.arange(2, 10, dtype=int)),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0, 1, 0.1),
        'weighting_function':weight_function
    }
  if weight_function == 'sigmoid':
    for data_group in qip_dict:
        for data_subset in qip_dict[data_group]:
            for qip in qip_dict[data_group][data_subset]:
                space.update({f'{data_subset}-{qip}-l':hp.uniform(f'{data_subset}-{qip}-l',0,1)})
                space.update({f'{data_subset}-{qip}-k':hp.uniform(f'{data_subset}-{qip}-k',0,10)})
                space.update({f'{data_subset}-{qip}-x_0':hp.uniform(f'{data_subset}-{qip}-x_0',0,10)})

  if weight_function == 'linear':
    for data_group in qip_dict:
        for data_subset in qip_dict[data_group]:
            for qip in qip_dict[data_group][data_subset]:
                space.update({f'{data_subset}-{qip}-m':hp.uniform(f'{data_subset}-{qip}-m',1,50)})
                space.update({f'{data_subset}-{qip}-b':hp.uniform(f'{data_subset}-{qip}-b',0,1)})

  if weight_function == 'direct':
    for data_group in qip_dict:
        for data_subset in qip_dict[data_group]:
            space.update({f'{data_subset}-weight':hp.uniform(f'{data_subset}-weight',0,1)})
  return space

#needs to be called inside a fmin function
def cv_mean_testset(indices_and_params:dict):
    '''vanilla cross validation with the loss returned being an average over all test sets
        all of data gets to be a test set once'''
    varity_data = indices_and_params["varity_data"]
    weight_func = indices_and_params["weighting_function"]
    errors_list = []
    kf = skm.KFold(n_splits=10, shuffle=True)
    parameter_dict = {"eta":indices_and_params['eta'], "gamma": indices_and_params['gamma'], "max_depth":indices_and_params['max_depth'], "min_child_weight":indices_and_params['min_child_weight'],
    "subsample":indices_and_params['subsample'], "colsample_bytree":indices_and_params['colsample_bytree'],"tree_method":'exact',"objective":"reg:logistic"}
    varity_featurelist = list(varity_data.feature_set)
    varity_r_data_unsplit_labelled = varity_data.data[varity_featurelist+['label']]
    #varity_r_data_unsplit_features_only = varity_r_data_unsplit_labelled[list_features].iloc[]
    #labels = varity_r_data_unsplit_labelled["label"]
    
    weights = wf.Weight(varity_data.data,varity_data.qip_dict)
    if weight_func == 'linear':
        weight_func_exec = weights.linear
    if weight_func == 'sigmoid':
        weight_func_exec = weights.sigmoid
    if weight_func == 'direct': 
        weight_func_exec = weights.direct
    else: raise Exception("no weight function provided")
    weights.fw_core_multiply_weight_vector_maker(varity_data.data, varity_data.qip_dict,indices_and_params,weight_func_exec,True)
    for train, test in kf.split(varity_r_data_unsplit_labelled):
        print(type(train))
        weights_train = weights.weights[train]
        weights_test = weights.weights[test]
        train = varity_r_data_unsplit_labelled.iloc[train.tolist()]
        test = varity_r_data_unsplit_labelled.iloc[test.tolist()]
        print(train.shape)

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

def nested_cv_():
    '''nested cross validation - find the best set of hyperparameters on outer train set then validate
    on outer loop test set. This adds a multiplicative factor on the number of times the model is fit. 
    '''
    pass

def core_set_finder(data: data.Dataloader_Varity) -> np.ndarray[int]:
    '''data -> Dataloader_Varity instance 
        this function returns a numpy array that contains all of the integer indices of core set members'''
    qip_dict = data.qip_dict
    core_sets = qip_dict['core'].keys()
    index_array = np.array([],dtype=int)
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
    weight_func = indices_and_params["weighting_function"]
    

    errors_list = []
    kf = skm.KFold(n_splits=5, shuffle=True)
    parameter_dict = {"eta":indices_and_params['eta'], "gamma": indices_and_params['gamma'], "max_depth":indices_and_params['max_depth'], "min_child_weight":indices_and_params['min_child_weight'],
    "subsample":indices_and_params['subsample'], "colsample_bytree":indices_and_params['colsample_bytree'],"tree_method":'exact',"objective":"reg:logistic"}
    varity_featurelist = list(varity_data.feature_set)
    varity_r_data_unsplit_labelled = varity_data.data[varity_featurelist+['label']]
    
    weights = wf.Weight(varity_data.data,varity_data.qip_dict)
    if weight_func == 'linear':
        weight_func_exec = weights.linear
    elif weight_func == 'sigmoid':
        weight_func_exec = weights.sigmoid
    elif weight_func == 'direct':
        weight_func_exec= weights.direct
    else: raise Exception("no weight function provided")

    weights.fw_core_multiply_weight_vector_maker(varity_data.data, varity_data.qip_dict,indices_and_params,weight_func_exec,True)

    #get core set indices for splitting only core instances
    indices_core_set = core_set_finder(varity_data)
    #create an array with integers ranging from 0 to num rows in training data so that we can create a train set  
    dummy_index = np.arange(varity_data.data.shape[0])
    for train, test in kf.split(indices_core_set):
        test_indices = indices_core_set[test] #get indices based on actual training data, train and test are indices into the core set only 
        train_indices = indices_core_set[train]
        #np.in1d compares two arrays and returns a boolean with anywhere elements in arg2 appear in arg1
        #a[np.in1d(a,b,invert=True)] gives you all elements in a that are NOT in b
        train_indices = dummy_index[np.in1d(dummy_index,test_indices,invert=True)]
        weights_train = weights.weights[train_indices]
        weights_test = weights.weights[test_indices]
  
        train = varity_r_data_unsplit_labelled.iloc[train_indices]
        test = varity_r_data_unsplit_labelled.iloc[test_indices]


        train_dataset = xgb.DMatrix(train[varity_featurelist],weight=weights_train, label=train["label"],feature_names=varity_featurelist)
        test_dataset = xgb.DMatrix(test[varity_featurelist],weight=weights_test, label=test["label"],feature_names=varity_featurelist)

        model = xgb.train(parameter_dict,train_dataset,num_boost_round=indices_and_params['num_boost_round'])

        predicted_labels = model.predict(test_dataset)
        #print(predicted_labels.shape)
        prior = prior_calc(test)
        error = aubprc(test["label"],predicted_labels,prior)
        #print(prior)
        errors_list.append(error)

    
    return {'loss':-1*mean(errors_list), 'status':STATUS_OK}

if __name__ == "__main__":
    varity_data = data.Dataloader_Varity("/Users/alirezarasoulzadeh/Desktop/reimplemented_varity/test_config.json")
    hp_dict = hp_space_builder_varity(varity_data.qip_dict,'direct')
    trials = Trials()
    hp_dict.update({"varity_data":varity_data})
    best = fmin(core_targeted_CV,
    space = hp_dict,
    algo=tpe.suggest,
    max_evals=10, trials=trials)
    sample = hyperopt.pyll.stochastic.sample(hp_dict)