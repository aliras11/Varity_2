'''testing facility for important Varity functions'''

import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import dataloader_varity as data
from weighting_functions import Weight


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
def test_all_weight_maker(data_subset: str, qip: str, test_data: data.Dataloader_Varity):
    '''test the function that assigns weights to all instances, regardless of whether they are a core or extra instances
    data_subset -> the data subset weights are being calculated for, for example, "core_clinvar_0
    qip -> quality informative property being used to calculate weights
    test_data -> loaded and processed training data'''
    
    weights = Weight(test_data.data, test_data.qip_dict)
    results =  weights.all_weight_maker(data_subset,qip,weights._sigmoid,(1,1,0),False)
    assert len(results) == test_data.data.shape[0], "Weight array length does not match number of rows in dataset provided"
    assert (results<=np.ones(len(results))).all(), "Some weights have values greater than 1!"
    assert (weights._sigmoid(test_data.data.loc[(test_data.data["set_name"] == data_subset)][qip].to_numpy(),1,1,0) == results[(test_data.data["set_name"] == data_subset)]).all(), "Weighting function miscalculation occured"



#test the function that assigns weights only to the extra instances, and automatically assigns fullweight to the core instances
#currently only supports testing the multiplicative weights
def test_core_fw_weight_maker(data_group: str,data_subset: str, qip: str, test_data: data.Dataloader_Varity):
    '''test the function that assigns weights to extra instances
    data_group -> extra or core
    data_subset -> the data subset weights are being calculated for, for example, "core_clinvar_0
    qip -> quality informative property being used to calculate weights
    test_data -> loaded and processed training data'''
    weights = Weight(test_data.data, test_data.qip_dict)
    results =  weights.fw_core_weight_maker(data_group,data_subset,qip,weights._sigmoid,(1,1,0),False)

    assert len(results) == test_data.data.shape[0], "Weight array length does not match number of rows in dataset provided"

    if data_group.lower() == 'core':
        assert (results == np.ones(test_data.data.shape[0])).all(), "Core instances should be give full weight"
    
    assert (results<=np.ones(len(results))).all(), "Some weights have values greater than 1!"
    
    if data_group.lower() =='extra':
        assert (weights._sigmoid(test_data.data.loc[(test_data.data["set_name"] == data_subset)][qip].to_numpy(),1,1,0) == results[(test_data.data["set_name"] == data_subset)]).all(), "Weighting function miscalculation occured"


if __name__ == "__main__":
    train_data = data.Dataloader_Varity("/Users/alirezarasoulzadeh/Desktop/reimplemented_varity/test_config.json")
    print(qip_dict_subset_extractor(train_data.qip_dict))
    print(test_all_weight_maker('core_clinvar_1','clinvar_review_star',train_data))
    print(test_core_fw_weight_maker('extra','extra_gnomad_high','neg_log_af',train_data))