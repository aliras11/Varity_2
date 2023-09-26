
import xgboost as xgb
import pandas as pd
import sklearn.preprocessing as skp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import sklearn.model_selection as skm
import sklearn.metrics as smm
import pickle
from statistics import mean
import numpy as np
import scipy.special as ssp
import random
from hyperopt.pyll.stochastic import sample



def fake_dataset_maker(qip_dict_test, size, seed=42):
  '''take in a qip_dict and create a testing dataset for validating varity functions'''
  data_dict_1 = {}
  data_dict_2 = {}
  variants = [f"var-{i}" for i in range(size)]

  random.seed(seed)

  random.shuffle(variants)
  data_dict_1.update({'variants':variants})
  data_dict_2.update({'variants':variants})
  data_group = list(qip_dict.keys())
  temp_list = []
  for data_subset in data_group:
    temp_list += list(qip_dict[data_subset].keys()) #dump datasubsets here

  random.seed(seed)
  random_set_sample = random.choices(temp_list,k=size)
  data_dict_1.update({'set_name':random_set_sample})
  data_dict_2.update({'set_name':random_set_sample})
  data_frame = pd.DataFrame(data_dict_1)
  qip_list = []

  for data_group in qip_dict:
    for data_subset in qip_dict[data_group]:

      bool_mask = (data_frame['set_name'] == data_subset)
      qip_array = np.empty((size,))
      qip_array.fill(np.nan)

      for qip in qip_dict[data_group][data_subset]:
        qip_array = np.empty((size,))
        qip_array.fill(np.nan)
        try:
          #np.random.seed(seed)
          data_dict_2[qip][bool_mask] = np.random.randint(100, size=np.sum(bool_mask.to_numpy()))
        except:
          qip_array[bool_mask] = np.random.randint(100, size=np.sum(bool_mask.to_numpy()))
          data_dict_2.update({f'{qip}':qip_array})

  random.seed(seed)
  label = random.choices([0,1],k=size)
  data_dict_2.update({'label':label})
  return pd.DataFrame(data_dict_2)


def sigmoid(x,l,k,x_0):
  '''calculates the value of a sigmoid function parametrized by the following arguments
      x -> pandas column/series of quality informative properties
      l -> dynamic range of sigmoid function, maximum value sigmoid is allowed to take
      k -> ascent rate of sigmoid
      x_0 -> right/left translation of sigmoid function
  '''
  return (l / (1 + np.exp(-k*(x-x_0))))

def weight_maker(train_data, data_group, data_subset, qip, weight_function, weight_function_args):

  ''' assign weights to one training set and QIP combination as per a pair-unique sigmoid function
      train_data -> varity training pandas df
      data_group -> core or extra (add-on) training set
      data_subset -> specific training set e.g. extra_clinvar_0_high
      qip -> qip to input into sigmoid function
      weight_function -> function to apply for weighting
      weight_function_args -> arguments to weight function, this makes sigmoid function unique to each call
  '''

  len_weight_array = train_data.shape[0]
  if data_group.lower() == "core":
    return np.ones((len_weight_array,))
  else:
    weight_array = np.ones((len_weight_array,))
    weight_mask = (train_data["set_name"] == data_subset)
    #TODO try making args dictionary based
    calculated_weights = train_data.loc[(train_data["set_name"] == data_subset)][qip].apply(weight_function, args=weight_function_args)
    calculated_weights = calculated_weights.to_numpy()
    weight_array[weight_mask] =  calculated_weights
    #weight_array = np.expand_dims(weight_array,axis=1) #make it 2D array with dims (n,1) so we can concatenate later
    return weight_array

def weight_vector_maker(train_data, qip_dict, args_dict, rebalance=True):

  '''take in a quality informative property dictionary (provided as a configuration json file) and assign weights to each
      feature QIP combination, returning a 1D numpy array representing weights
      train_data -> varity training pandas df
      qip_dict -> dictionary relating data subsets to QIPs
      args_dict -> hyperopt space dict from which we extract QIP specific sigmoid parameters'''

  print(args_dict.keys())
  mul_weight_vector = np.ones((train_data.shape[0],))
  #weights_matrix = pd.DataFrame()
  for data_group in qip_dict:
    for data_subset in qip_dict[data_group]:
      for qip in qip_dict[data_group][data_subset]:
        x0 = args_dict[f'{data_subset}-{qip}-x_0']
        k = args_dict[f'{data_subset}-{qip}-k']
        l = args_dict[f'{data_subset}-{qip}-l']
        weight_args = (x0, k, l)
        print(f"{data_group} - {data_subset} - {qip}")
        temp_weight_vector = weight_maker(train_data,data_group,data_subset,qip,sigmoid,weight_args)
        #weights_matrix[f"{data_group} - {data_subset} - {qip}"] = temp_weight_vector
        mul_weight_vector = np.multiply(mul_weight_vector,temp_weight_vector)
        print(mul_weight_vector.shape)
        #added as column shouldve probably been rows

  if rebalance:
    #balancing performed here
    neg_samples = (train_data["label"] == 0)
    pos_samples = (train_data["label"] == 1)

    if round(mul_weight_vector[neg_samples].sum(),1) > round(mul_weight_vector[pos_samples].sum(),1):
        print("negative examples")
        print(len(mul_weight_vector[neg_samples]))
        print(mul_weight_vector[neg_samples].sum())
        balance_ratio = mul_weight_vector[pos_samples].sum()/mul_weight_vector[neg_samples].sum()
        mul_weight_vector[neg_samples] = mul_weight_vector[neg_samples]*balance_ratio
        print(mul_weight_vector.shape)
        print(f"{round(mul_weight_vector[neg_samples].sum(),1)} - {round(mul_weight_vector[pos_samples].sum(),1)}")

    if round(mul_weight_vector[neg_samples].sum(),1) < round(mul_weight_vector[pos_samples].sum(),1):
        print("positive examples")
        print(mul_weight_vector[pos_samples].sum())

        balance_ratio = mul_weight_vector[neg_samples].sum()/mul_weight_vector[pos_samples].sum()
        mul_weight_vector[pos_samples] = mul_weight_vector[pos_samples]*balance_ratio

    if mul_weight_vector[neg_samples].sum() == mul_weight_vector[pos_samples].sum():
        pass
  return mul_weight_vector

def hp_space_builder_varity(qip_dict):
  space = space = {
        'num_boost_round':hp.choice('num_boost_round', np.arange(2, 10, dtype=int)),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0, 1, 0.1)
    }
  for data_group in qip_dict:
    for data_subset in qip_dict[data_group]:
      for qip in qip_dict[data_group][data_subset]:
        space.update({f'{data_subset}-{qip}-l':hp.uniform(f'{data_subset}-{qip}-l',0,10)})
        space.update({f'{data_subset}-{qip}-k':hp.uniform(f'{data_subset}-{qip}-k',0,10)})
        space.update({f'{data_subset}-{qip}-x_0':hp.uniform(f'{data_subset}-{qip}-x_0',0,10)})
  return space


if __name__ == "__main__":
  qip_dict = {'core':{'core_clinvar_0':["clinvar_review_star",'neg_log_af'],
                    'core_clinvar_1':["clinvar_review_star",'neg_log_af']},

  'extra':{'extra_clinvar_0_high':["clinvar_review_star",'neg_log_af'],
          'extra_clinvar_1':["clinvar_review_star",'neg_log_af'],
          'extra_gnomad_high':['neg_log_af','gnomAD_exomes_nhomalt'],
          'extra_gnomad_low':['neg_log_af','gnomAD_exomes_nhomalt'],
          'extra_hgmd':['neg_log_af'],
          'extra_humsavar_0_high':['neg_log_af'],
          'extra_humsavar_0_low':['neg_log_af'],
          'extra_humsavar_1':['neg_log_af'],
          'extra_mave_0':['mave_label_confidence','accessibility'],
          'extra_mave_1':['mave_label_confidence','accessibility']}
          }


  a = fake_dataset_maker(qip_dict, 1000, seed=42)
  space = hp_space_builder_varity(qip_dict)
  example_space = sample(space)
  test_weight_vector = weight_vector_maker(a,qip_dict,example_space)
  test_weight_vector