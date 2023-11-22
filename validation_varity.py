import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from pathlib import Path
import pickle 
import numpy as np
import pandas as pd
import dataloader_varity as data
import weighting_functions as wf
import load_model as load
import argparse
from hp_tuning_skeleton import aubprc

def prior_calc(data_frame):

    ''' calculates the ratio of positive to negative examples, p(Y=1)
    data_frame -> pandas data frame, must have a label column with 0 or 1 values
    '''
    freq_dict = data_frame["denovo_label"].value_counts().to_dict()
    if 1 in freq_dict.keys() and 0 in freq_dict.keys():
        negative = freq_dict[0]
        positive = freq_dict[1]
        prior = positive/(positive+negative) #calculate prior probability of pathogeneticity 

    elif 1 in freq_dict.keys() and 0 not in freq_dict.keys(): #if there are only pathogenic/positive labels 
        prior = 1 #prior if there are no positive cases

    else: #if there are no pathogenic/positive labels 
        prior = 0
    
    return prior

def main(train_config_path:str,valid_config_path: str,weight_function:str, trial_path:str) -> int:
    valid_data = data.Validationloader_Varity(valid_config_path)
    #train_data = data.Dataloader_Varity(train_config_path)
    loader = load.Best_Model_Builder(trial_path,train_config_path)
    model = loader.build_train_model(weight_function)
    valid_dataset = xgb.DMatrix(valid_data.data[list(valid_data.feature_set)], label=valid_data.data["denovo_label"],feature_names=list(valid_data.feature_set))
    predict = model.predict(valid_dataset)
    prior = prior_calc(valid_data.data)
    objective = aubprc(valid_data.data["denovo_label"],predict,prior)
    return objective

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", help="training data configuration file, needed to train a model for a given set of hyperparameters")
    parser.add_argument("valid_config", help="validation data configuration file, needed to extract relevant features")
    parser.add_argument("weight_function", help="function to use when calculating weights form quality informative properties")
    parser.add_argument("trial_path", help="path to hyperopt trial object we want to extract hp's from")
    args = parser.parse_args()
    print(main(args.train_config, args.valid_config,args.weight_function,args.trial_path))