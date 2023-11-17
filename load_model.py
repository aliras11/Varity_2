import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from pathlib import Path
import pickle 
import numpy as np
import pandas as pd
import dataloader_varity as data
import weighting_functions as wf


class Best_Model_Builder():
    def __init__(self,trial_path: str, config_path:str):
        '''path -> abs path to pickled trials object '''
        self.trial_path = Path(trial_path) 
        self.config_path = Path(config_path)
        self.best_trial = self.get_best_hps()
        self.trials_object = self.get_trials_dict()
        #make a dataloader class instance 

    def get_best_hps(self):
        with open(self.trial_path, "rb") as hp_tunedata:
            trials_dict = pickle.load(hp_tunedata)
            trials_dict = trials_dict.best_trial
        return trials_dict

    def get_trials_dict():
        pass
    def get_weights(self,train_data,qip_dict):
        args_dict = self.best_trial['misc']['vals']
        varity_data = data.Dataloader_Varity(str(self.config_path))
        weights = wf.Weight(varity_data.data,varity_data.qip_dict)
        weights.fw_core_multiply_weight_vector_maker(train_data,qip_dict, args_dict,weights.direct, rebalance=True)
        return weights.weights

if __name__ == "__main__":
    a = Best_Model_Builder('/Users/alirezarasoulzadeh/Desktop/reimplemented_varity/results_trials_folder/direct_trial_object_hp_tuning_2023-11-17_15:31:29.422267.pkl','test_config.json')
    b = data.Dataloader_Varity("/Users/alirezarasoulzadeh/Desktop/reimplemented_varity/test_config.json")
    print(a.get_weights(b.data,b.qip_dict))
    print(a.best_trial)

