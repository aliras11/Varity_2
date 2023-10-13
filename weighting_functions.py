'''a library of functions used in the Varity framework for weighting instances in the XGBoost algorithm'''

import numpy as np
import pandas as pd
import dataloader_varity as data

class Weight():
    def __init__(self,data,qip_dict):
        self.data = data
        self.qip_dict = qip_dict
        self.weights = None
    
    def _sigmoid(self,x,l,k,x_0):
        '''calculates the value of a sigmoid function parametrized by the following arguments
        x -> pandas column/series of quality informative properties
        l -> dynamic range of sigmoid function, maximum value sigmoid is allowed to take
        k -> ascent rate of sigmoid
        x_0 -> right/left translation of sigmoid function'''
        return (l/(1+np.exp(-1*k*(x-x_0))))


    #assigns full weight to any core training group, helper function, has a return value does not set attribute directly
    def fw_core_weight_maker(self, data_group, data_subset, qip, weight_function, weight_function_args,additive):

        ''' assign weights to one training set and QIP combination as per a pair-unique sigmoid function
            train_data -> varity training pandas df
            data_group -> core or extra (add-on) training set, string
            data_subset -> specific training set e.g. extra_clinvar_0_high, string
            qip -> qip to input into sigmoid function, pandas series
            weight_function -> function to apply for weighting
            weight_function_args -> arguments to weight function, this makes sigmoid function unique to each call
            additive -> boolean, indicating how weights will later be combined across different QIPs
        '''
        len_weight_array = self.data.shape[0]
        if data_group.lower() == "core":
            return np.ones((len_weight_array,)) 
        else:

            if additive:
                weight_array = np.zeros((len_weight_array,))
            else:
                weight_array = np.ones((len_weight_array,))

            weight_mask = (self.data["set_name"] == data_subset)

            calculated_weights = self.data.loc[(self.data["set_name"] == data_subset)][qip].apply(weight_function, args=weight_function_args)
            calculated_weights = calculated_weights.to_numpy()

            weight_array[weight_mask] =  calculated_weights
            #weight_array = np.expand_dims(weight_array,axis=1) #make it 2D array with dims (n,1) so we can concatenate later

            return weight_array

    #assign weights to all instances using a weighting function, including core
    def all_weight_maker(self, data_subset, qip, weight_function, weight_function_args,additive):

        ''' assign weights to one training set and QIP combination as per a pair-unique sigmoid function
        data_subset -> specific training set e.g. extra_clinvar_0_high, string
        qip -> qip to input into sigmoid function, pandas series
        weight_function -> function to apply for weighting
        weight_function_args -> arguments to weight function, this makes sigmoid function unique to each call
        additive -> boolean, indicating how weights will later be combined across different QIPs
        '''
        len_weight_array = self.data.shape[0]
        if additive:
            weight_array = np.zeros((len_weight_array,))
        else:
            weight_array = np.ones((len_weight_array,))
        weight_mask = (self.data["set_name"] == data_subset)
        calculated_weights = self.data.loc[(self.data["set_name"] == data_subset)][qip].apply(weight_function, args=weight_function_args)
        calculated_weights = calculated_weights.to_numpy()
        weight_array[weight_mask] =  calculated_weights
        #weight_array = np.expand_dims(weight_array,axis=1) #make it 2D array with dims (n,1) so we can concatenate later

        return weight_array
    
    #runs the fw_core_weight_maker function for assigning weights
    def fw_core_multiply_weight_vector_maker(self,train_data,qip_dict, args_dict, rebalance=True):

        '''take in a quality informative property dictionary (provided as a configuration json file) and assign weights to each
            feature QIP combination, returning a 1D numpy array representing weights
            train_data -> varity training pandas df
            qip_dict -> dictionary relating data subsets to QIPs
            args_dict -> hyperopt space dict from which we extract QIP specific sigmoid parameters'''

        mul_weight_vector = np.ones((train_data.shape[0],))
        #weights_matrix = pd.DataFrame()
        for data_group in qip_dict:
            for data_subset in qip_dict[data_group]:
                for qip in qip_dict[data_group][data_subset]:
                    x0 = args_dict[f'{data_subset}-{qip}-x_0']
                    k = args_dict[f'{data_subset}-{qip}-k']
                    l = args_dict[f'{data_subset}-{qip}-l']
                    weight_args = (l, k, x0)
                    print(f"{data_group} - {data_subset} - {qip}")
                    temp_weight_vector = self.fw_core_weight_maker(data_group,data_subset,qip,self._sigmoid,weight_args,False)
                    #weights_matrix[f"{data_group} - {data_subset} - {qip}"] = temp_weight_vector
                    mul_weight_vector = np.multiply(mul_weight_vector,temp_weight_vector)
                    print(mul_weight_vector.shape)
                    #added as column shouldve probably been rows

        if rebalance:
            #balancing performed here
            neg_samples = (train_data["label"] == 0)
            pos_samples = (train_data["label"] == 1)

            if round(mul_weight_vector[neg_samples].sum(),1) > round(mul_weight_vector[pos_samples].sum(),1):

                balance_ratio = mul_weight_vector[pos_samples].sum()/mul_weight_vector[neg_samples].sum()
                mul_weight_vector[neg_samples] = mul_weight_vector[neg_samples]*balance_ratio

            if round(mul_weight_vector[neg_samples].sum(),1) < round(mul_weight_vector[pos_samples].sum(),1):

                balance_ratio = mul_weight_vector[neg_samples].sum()/mul_weight_vector[pos_samples].sum()
                mul_weight_vector[pos_samples] = mul_weight_vector[pos_samples]*balance_ratio

            if np.isclose(mul_weight_vector[neg_samples].sum(),mul_weight_vector[pos_samples].sum()):
                self.weights = mul_weight_vector

            else:
                raise Exception("Unable to balance weights")

    #runs the all_weight_maker function for assigning weights
    def aw_multiply_weight_vector_maker(self,train_data,qip_dict, args_dict, rebalance=True):
        
        '''take in a quality informative property dictionary (provided as a configuration json file) and assign weights to each
            feature QIP combination, returning a 1D numpy array representing weights
            train_data -> varity training pandas df
            qip_dict -> dictionary relating data subsets to QIPs
            args_dict -> hyperopt space dict from which we extract QIP specific sigmoid parameters'''

        mul_weight_vector = np.ones((train_data.shape[0],))
        #weights_matrix = pd.DataFrame()
        for data_group in qip_dict:
            for data_subset in qip_dict[data_group]:
                for qip in qip_dict[data_group][data_subset]:
                    x0 = args_dict[f'{data_subset}-{qip}-x_0']
                    k = args_dict[f'{data_subset}-{qip}-k']
                    l = args_dict[f'{data_subset}-{qip}-l']
                    weight_args = (l, k, x0)
                    #additive set to false because this function only multiplies weights
                    temp_weight_vector = self.all_weight_maker(data_subset,qip,self._sigmoid,weight_args,False)
                    
                    mul_weight_vector = np.multiply(mul_weight_vector,temp_weight_vector)
                    print(mul_weight_vector.shape)
                    #added as column shouldve probably been rows
        if rebalance:
        #balancing performed here
            neg_samples = (train_data["label"] == 0)
            pos_samples = (train_data["label"] == 1)

            if round(mul_weight_vector[neg_samples].sum(),1) > round(mul_weight_vector[pos_samples].sum(),1):

                balance_ratio = mul_weight_vector[pos_samples].sum()/mul_weight_vector[neg_samples].sum()
                mul_weight_vector[neg_samples] = mul_weight_vector[neg_samples]*balance_ratio

            if round(mul_weight_vector[neg_samples].sum(),1) < round(mul_weight_vector[pos_samples].sum(),1):

                balance_ratio = mul_weight_vector[neg_samples].sum()/mul_weight_vector[pos_samples].sum()
                mul_weight_vector[pos_samples] = mul_weight_vector[pos_samples]*balance_ratio

            if np.isclose(mul_weight_vector[neg_samples].sum(),mul_weight_vector[pos_samples].sum()):
                 self.weights = mul_weight_vector
            else:
                raise Exception("Unable to balance weights")


if __name__ == "__main__":
    varity_data = data.Dataloader_Varity("/Users/alirezarasoulzadeh/Desktop/reimplemented_varity/test_config.json")
    weights = Weight(varity_data.data,varity_data.qip_dict)
    args_dict = {}
    args_dict = testing.test_hp_space_builder(varity_data.qip_dict)
    weights.fw_core_multiply_weight_vector_maker(varity_data.data,varity_data.qip_dict,args_dict)
    print(args_dict)
    
