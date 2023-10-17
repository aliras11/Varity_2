'''module defining different types of hyperparameter tuning loops'''
import xgboost as xgb
import sklearn.model_selection as skm
from statistics import mean
from dataloader_varity import *

a = Dataloader_Varity()
dataset = a.load_csv("/Users/alirezarasoulzadeh/Desktop/reimplemented_varity/config.json")

def varity_objective_CV(hyperopt_tuning_dict):
  '''
  hyperopt_tuning_dict -> dictionary provided by the hyperopt fmin function containing the sampled hp values

  '''
  global dataset
  errors_list = []
  kf = skm.KFold(n_splits=10)
  parameter_dict = {"eta":hyperopt_tuning_dict['eta'], "gamma": hyperopt_tuning_dict['gamma'], "max_depth":hyperopt_tuning_dict['max_depth'], "min_child_weight":hyperopt_tuning_dict['min_child_weight'],
  "subsample":hyperopt_tuning_dict['subsample'], "colsample_bytree":hyperopt_tuning_dict['colsample_bytree'],"tree_method":'gpu_hist',"objective":"reg:logistic"}
  varity_r_data_unsplit_labelled = dataset[list_features_with_label]
  #varity_r_data_unsplit_features_only = varity_r_data_unsplit_labelled[list_features].iloc[]
  #labels = varity_r_data_unsplit_labelled["label"]

  weights = (dataset,qip_dict,hyperopt_tuning_dict)
  for train, test in kf.split(varity_r_data_unsplit_labelled):
    print(type(train))
    weights_train = weights[train]
    weights_test = weights[test]
    train = varity_r_data_unsplit_labelled.iloc[train.tolist()]
    test = varity_r_data_unsplit_labelled.iloc[test.tolist()]
    print(train.shape)

    train_dataset = xgb.DMatrix(train[list_features],weight=weights_train, label=train["label"],feature_names=list_features)
    test_dataset = xgb.DMatrix(test[list_features],weight=weights_test, label=test["label"],feature_names=list_features)

    model = xgb.train(parameter_dict,train_dataset,num_boost_round=hyperopt_tuning_dict['num_boost_round'])

    predicted_labels = model.predict(test_dataset)
    print(predicted_labels.shape)
    prior = prior_calc(test)
    error = aubprc(test["label"],predicted_labels,prior)
    print(prior)
    errors_list.append(error)

  print(errors_list)
  return {'loss':-1*mean(errors_list), 'status':STATUS_OK}
