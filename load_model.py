import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from pathlib import Path
import pickle 
import numpy as np
import pandas as pd
import dataloader_varity as data
import weighting_functions as wf


class Best_Model_Builder():
    def __init__(self,trial_path: str, config_path:str, model_path: str=""):
        '''path -> abs path to pickled trials object '''
        self.trial_path = Path(trial_path) 
        self.config_path = Path(config_path)
        self.model_path = Path(model_path)
        self.best_trial = self.get_best_hps()
        self.trials_object = self.get_trials_dict()
        #make a dataloader class instance 

    def get_best_hps(self):
        with open(self.trial_path, "rb") as hp_tunedata:
            trials_dict = pickle.load(hp_tunedata)
            trials_dict = trials_dict.best_trial['misc']['vals'] #structure of trials object is as such
        for hp in trials_dict.keys():
           trials_dict[hp] = trials_dict[hp][0] #unpack lists, trials object is a dict of one dimensional lists so take first element
        return trials_dict

    def get_trials_dict(self):
        with open(self.trial_path, "rb") as hp_tunedata:
            trials_dict = pickle.load(hp_tunedata)
        return trials_dict

    def get_weights(self,weight_function:str):
        args_dict = self.best_trial
        varity_data = data.Dataloader_Varity(str(self.config_path))
        weights = wf.Weight(varity_data.data,varity_data.qip_dict)
        if weight_function == "direct":
            weights.fw_core_multiply_weight_vector_maker(varity_data.data,varity_data.qip_dict, args_dict,weights.direct, rebalance=True)
        elif weight_function == "sigmoid":
            weights.fw_core_multiply_weight_vector_maker(varity_data.data,varity_data.qip_dict, args_dict,weights.sigmoid, rebalance=True)
        elif weight_function == "linear":
            weights.fw_core_multiply_weight_vector_maker(varity_data.data,varity_data.qip_dict, args_dict,weights.linear, rebalance=True)
        elif weight_function == "all_ones":
            weights.fw_core_multiply_weight_vector_maker(varity_data.data,varity_data.qip_dict, args_dict,weights.all_ones, rebalance=True)
        else:
            raise Exception("Not Implemented")
        return weights.weights

    def build_train_model(self,weight_function):
        '''build and train model based on best trial in a given trials object'''
        best_trial_dict = self.get_best_hps()
        varity_data = data.Dataloader_Varity(str(self.config_path))
        varity_featurelist = list(varity_data.feature_set)
        varity_data_unsplit_labelled = varity_data.data[varity_featurelist+['label']]
        weights = self.get_weights(weight_function)
        train_dataset = xgb.DMatrix(varity_data_unsplit_labelled[varity_featurelist],weight=weights, label=varity_data_unsplit_labelled["label"],feature_names=varity_featurelist)
        model = xgb.train(best_trial_dict,train_dataset,num_boost_round=best_trial_dict['num_boost_round'])
        return model
    
if __name__ == "__main__":
    a = Best_Model_Builder('/Users/alirezarasoulzadeh/Desktop/reimplemented_varity/results_trials_folder/direct_trial_object_hp_tuning_2023-11-17_15:31:29.422267.pkl','test_config.json')
    b = data.Dataloader_Varity("/Users/alirezarasoulzadeh/Desktop/reimplemented_varity/test_config.json")
    print(a.get_weights('direct'))
    print(a.best_trial)

