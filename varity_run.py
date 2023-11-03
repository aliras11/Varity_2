'''main script from which a varity training/hp_tuning session is ran'''

import pickle 
import numpy as np
import pandas as pd
import dataloader_varity as data
import hp_tuning_skeleton as hp_tune
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from pathlib import Path
import datetime

def main() -> int:
    deleteme = 5556
    '''main function that executes a hyperparameter tuning session
        1. create and instantiate a varity dataloader instance 
        2. create a trials object for hyperopt to dump results into 
        3. pickle serialize trials object'''
    cp = Path.cwd()/Path('results_trials_folder')
    cp.mkdir(parents=True, exist_ok=True)
    varity_data = data.Dataloader_Varity("/Users/alirezarasoulzadeh/Desktop/reimplemented_varity/test_config.json")
    trials = Trials()
    hp_dict = hp_tune.hp_space_builder_varity(varity_data.qip_dict)
    hp_dict.update({'varity_data':varity_data})
    print(type(hp_dict))
    print(hp_dict.keys())
    best = fmin(hp_tune.core_targeted_CV,
    space = hp_dict,
    algo=tpe.suggest,
    max_evals=10, trials=trials)
    p = cp/Path(f"trial_object_hp_tuning_{deleteme}")
    with p.open(mode="wb") as fp:
        pickle.dump(trials, fp)

    return 0
if __name__ == "__main__":
    main()