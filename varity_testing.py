'''testing facility for important Varity functions'''

import numpy as np
import pandas as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def test_hp_space_builder(qip_dict):
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


