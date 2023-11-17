import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from pathlib import Path
import pickle 
import numpy as np
import pandas as pd
import dataloader_varity as data
import weighting_functions as wf


class Best_Model_Builder():
    def __init__(self,path: str):
        '''path -> abs path to pickled trials object '''
        self.path = Path(path) 
        self.best_trial = self.get_best_hps()
        #make a dataloader class instance 

    def get_best_hps(self):
        with open(self.path, "rb") as hp_tunedata:
            trials_dict = pickle.load(hp_tunedata)
            trials_dict = trials_dict.best_trial
        return trials_dict

    def get_weights(self,train_data,qip_dict):
        args_dict = self.best_trial['misc']['vals']
        weights = wf.Weight()
        fw_core_multiply_weight_vector_maker(train_data,qip_dict, args_dict, rebalance=True)
        return weights

if __name__ == "__main__":
    a = Best_Model_Builder('/Users/alirezarasoulzadeh/Downloads/hp_tuning_varity_300_nov7.pkl')
    b = data.Dataloader_Varity("/Users/alirezarasoulzadeh/Desktop/reimplemented_varity/test_config.json")
    print(a.get_weights(b.data,b.qip_dict))
    print(a.best_trial)

