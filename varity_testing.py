'''testing facility for important Varity functions'''

import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import dataloader_varity as data
from weighting_functions import Weight
import math 


def test_hp_space_builder(qip_dict: dict) -> dict:
    '''used to generate test hyperparameter dictionaries that mimic hyperopt search spaces
    without the need to do any sampling which would introduce stochasticity. Values are hardcoded '''
    space = {
        'num_boost_round':7,
        'eta':0.5,
        'max_depth':12,
        'min_child_weight': 1,
        'subsample': 0.5,
        'gamma': 0.5,
        'colsample_bytree': 0.5
    }
    for data_group in qip_dict:
        for data_subset in qip_dict[data_group]:
            for qip in qip_dict[data_group][data_subset]:
                space.update({f'{data_subset}-{qip}-l':1})
                space.update({f'{data_subset}-{qip}-k':1})
                space.update({f'{data_subset}-{qip}-x_0':0})
    return space

#helper function that returns data subsets associated with a qip_dict,
def qip_dict_subset_extractor(qip_dict: dict) -> list:
    list_data_subset = []
    for data_group in qip_dict:
        for data_subset in qip_dict[data_group]:
            list_data_subset.append(data_subset)
    return list_data_subset
            



#test the function that assigns weights to all instances, regardless of whether they are a core or extra instances
#currently only supports testing the multiplicative weights
def all_weight_maker_test(data_subset: str, qip: str, test_data: data.Dataloader_Varity) -> None:
    '''test the function that assigns weights to all instances, regardless of whether they are a core or extra instances
    data_subset -> the data subset weights are being calculated for, for example, "core_clinvar_0
    qip -> quality informative property being used to calculate weights
    test_data -> loaded and processed training data'''
    
    weights = Weight(test_data.data, test_data.qip_dict)
    results =  weights.all_weight_maker(data_subset,qip,weights.sigmoid,(2,1,5),False)
    assert len(results) == test_data.data.shape[0], "Weight array length does not match number of rows in dataset provided"
    assert (results<=np.ones(len(results))).all(), "Some weights have values greater than 1!"
    assert (weights.sigmoid(test_data.data.loc[(test_data.data["set_name"] == data_subset)][qip].to_numpy(),2,1,5) == results[(test_data.data["set_name"] == data_subset)]).all(), "Weighting function miscalculation occured"



#test the function that assigns weights only to the extra instances, and automatically assigns fullweight to the core instances
#currently only supports testing the multiplicative weights
def core_fw_weight_maker_test(data_group: str,data_subset: str, qip: str, test_data: data.Dataloader_Varity) -> None:
    '''test the function that assigns weights to extra instances
    data_group -> extra or core
    data_subset -> the data subset weights are being calculated for, for example, "core_clinvar_0
    qip -> quality informative property being used to calculate weights
    test_data -> loaded and processed training data'''
    weights = Weight(test_data.data, test_data.qip_dict)
    results =  weights.fw_core_weight_maker(data_group,data_subset,qip,weights.sigmoid,(1,1,0),False)

    assert len(results) == test_data.data.shape[0], "Weight array length does not match number of rows in dataset provided"

    if data_group.lower() == 'core':
        assert (results == np.ones(test_data.data.shape[0])).all(), "Core instances should be give full weight"
    
    assert (results<=np.ones(len(results))).all(), "Some weights have values greater than 1!"

    if data_group.lower() =='extra':
        assert (weights.sigmoid(test_data.data.loc[(test_data.data["set_name"] == data_subset)][qip].to_numpy(),1,1,0) == results[(test_data.data["set_name"] == data_subset)]).all(), "Weighting function miscalculation occured"


def fw_multiply_tester(test_data: data.Dataloader_Varity, qip_dict: dict, args_dict: dict, rebalance: bool) -> None:
    '''test multiplicative form of final weight vector generating function, where the final form is the 
    result of the QIP dict provided being fully processed and the weights for each data subset-QIP pair
    being multiplied yielding a final weight vector
    test_data -> pandas data frame, accessed as an attribute of Dataloader class
    qip_dict -> dictionary of quality informative property and data subset pairs 
    args_dict -> argument dictionary for weighting function, in this case this a stand in for hyperopt search space
    rebalance -> if weights in function being tested should be rebalanced between positive and negative instances
    '''
    weights = Weight(test_data.data, test_data.qip_dict)
    weights.fw_core_multiply_weight_vector_maker(test_data.data,qip_dict, args_dict, rebalance=True) #returns none
    results = weights.weights 
    count_pos = sum(list((test_data.data["label"] == 1)))
    assert len(results) == test_data.data.shape[0], "Weight array length does not match number of rows in dataset provided"
    assert count_pos > 0, "Need at least one positive instance to check weight balancing"
    print(f'{count_pos} number of positive instances in data')
    assert math.isclose(results[(test_data.data["label"] == 1)].sum(),results[(test_data.data["label"] == 0)].sum())
    print(results)

#TODO write a function to verify sigmoid is working as expected
# TODO write a function to verify the weight assignment for every QIP  
if __name__ == "__main__":
    train_data = data.Dataloader_Varity("/Users/alirezarasoulzadeh/Desktop/reimplemented_varity/test_config.json")
    print(qip_dict_subset_extractor(train_data.qip_dict))
    print(all_weight_maker_test('core_clinvar_1','clinvar_review_star',train_data))
    print(core_fw_weight_maker_test('extra','extra_gnomad_high','neg_log_af',train_data))
    args_dict = test_hp_space_builder(train_data.qip_dict)
    fw_multiply_tester(train_data,train_data.qip_dict,args_dict ,True)