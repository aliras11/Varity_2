'''a library of functions used in the Varity framework for weighting instances in the XGBoost algorithm'''

import numpy as np
import pandas as pd
import dataloader_varity as data

class Weight():
    def __init__(self,data,qip_dict):
        self.data = data #change this to take a dataloader instance instead
        self.qip_dict = qip_dict
        self.weights = None
    
    def sigmoid(self,x,l,k,x_0):
        '''calculates the value of a sigmoid function parametrized by the following arguments
        x -> pandas column/series of quality informative properties
        l -> dynamic range of sigmoid function, maximum value sigmoid is allowed to take
        k -> ascent rate of sigmoid
        x_0 -> right/left translation of sigmoid function'''
        return (l/(1+np.exp(-1*k*(x-x_0))))

    def linear(self,x,m,b):
        '''linear weighting function with maximum clipped at 1
        argmument names based y = mx +b convention
        x -> pandas column/series of quality informative properties'''
        weight = np.multiply(x,m)
        weight += float(b)
        weight = np.clip(weight,0,1)
        return weight
    
    def direct(self,x,weight,throwaway):#throwaway is needed to because we need a tuple as args in the .apply
        '''allows bayesian optmizing to directly choose the weight for each datasubet
        x -> pandas column/series, this argument is needed to make the .apply method to work even though it is not needed
        weight -> weight value assigned'''
        weight = np.multiply(x,0)+weight
        return weight

    def all_ones(self,x,weight):
        weight = np.multiply(x,0)+weight
        return weight

    def  weight_arg_extractor(self,args_dict,weight_function,data_subset,qip):
        '''takes in a hyperopt space and extracts the arguments for the relevant weighting function,
        recall different weighting functions are parametrized differently. User must make sure the args_dict has 
        the correct values for the given weight function 
        args_dict -> hyperopt space dict 
        weight_function -> weight function (callable)
        data_subset -> named part of data, found in second level of qip dict
        qip -> quality informative property'''

        if weight_function.__name__ == 'linear': #extract name of weight function
            return (args_dict[f'{data_subset}-{qip}-m'],args_dict[f'{data_subset}-{qip}-b'])

        if weight_function.__name__ == 'sigmoid':
            return (args_dict[f'{data_subset}-{qip}-l'],
            args_dict[f'{data_subset}-{qip}-k'],args_dict[f'{data_subset}-{qip}-x_0'])
                    
        if weight_function.__name__ == 'direct':
            return (args_dict[f'{data_subset}-weight'],0)

        if weight_function.__name__ == 'all_ones':
            return (1,)




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
    def fw_core_multiply_weight_vector_maker(self,train_data,qip_dict,args_dict,weight_function ,rebalance=True):

        '''take in a quality informative property dictionary (provided as a configuration json file) and assign weights to each
            feature QIP combination, returning a 1D numpy array representing weights
            train_data -> varity training pandas df
            qip_dict -> dictionary relating data subsets to QIPs
            args_dict -> hyperopt space dict from which we extract QIP specific sigmoid parameters
            weight_function -> function to use for calculating weights'''

        mul_weight_vector = np.ones((train_data.shape[0],))
        #weights_matrix = pd.DataFrame()
        for data_group in qip_dict:
            for data_subset in qip_dict[data_group]:
                for qip in qip_dict[data_group][data_subset]:

                    weight_args = self.weight_arg_extractor(args_dict,weight_function,data_subset,qip)
                    temp_weight_vector = self.fw_core_weight_maker(data_group,data_subset,qip,weight_function,weight_args,False)
                    #weights_matrix[f"{data_group} - {data_subset} - {qip}"] = temp_weight_vector
                    mul_weight_vector = np.multiply(mul_weight_vector,temp_weight_vector)
                    #print(mul_weight_vector.shape)
                    #added as column shouldve probably been rows
        self.weights = mul_weight_vector
        self.weight_geo_mean_apply()
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
                print("balanced_weights")
                self.weights = mul_weight_vector

            else:
                raise Exception("Unable to balance weights")

    #runs the all_weight_maker function for assigning weights
    def aw_multiply_weight_vector_maker(self,train_data,qip_dict, args_dict, rebalance=True):
        #doesnt support other weighting functions yet 
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
                    temp_weight_vector = self.all_weight_maker(data_subset,qip,self.sigmoid,weight_args,False)
                    
                    mul_weight_vector = np.multiply(mul_weight_vector,temp_weight_vector)
                    #added as column shouldve probably been rows
        self.weights = mul_weight_vector
        self.weight_geo_mean_apply()
        if rebalance:
        #balancing performed here
            neg_samples = (train_data["label"] == 0)
            pos_samples = (train_data["label"] == 1)
            #TODO delete me 
            print(f'{neg_samples.shape[0]}_{pos_samples.shape[0]}_negative count to positive count')

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

    def count_qips(self,qip_dict: dict)-> dict:
        '''count the number of QIPs associated with each data_subset, e.g. "core_clinvar_0" '''
        qip_count = dict()
        for data_group in qip_dict:
            for data_subset in qip_dict[data_group]:
                count_qips = len(qip_dict[data_group][data_subset])
                temp_dict = {data_subset:count_qips}
                qip_count.update(temp_dict)
        return qip_count
    
    def geo_mean(self,weight_array, number: int) -> float: 
        '''find the geomtric mean of a weight that would have been the result of multiplying (number) values'''
        return np.power(weight_array,(1.0/number))

    def weight_geo_mean_apply(self)->None:
        '''apply the geo mean function to weights depending on how many QIPs are associated with a certain datasubset
        use only when weights are combined in a multiplicative way'''
        count_qip_dict = self.count_qips(self.qip_dict)
        if self.weights.any: #check if a weight array exists
            for data_subset in count_qip_dict:
                subset_mask = (self.data["set_name"] == data_subset)
                #print(self.weights[subset_mask])
                self.weights[subset_mask] = self.geo_mean(self.weights[subset_mask],count_qip_dict[data_subset])
                #print(self.weights[subset_mask])
        else:
            raise Exception("No weights calculated")

if __name__ == "__main__":
    varity_data = data.Dataloader_Varity("/Users/alirezarasoulzadeh/Desktop/reimplemented_varity/test_config.json")
    weights = Weight(varity_data.data,varity_data.qip_dict)
    from varity_testing import test_hp_space_builder
    args = test_hp_space_builder(varity_data.qip_dict,'direct')
    weights.fw_core_multiply_weight_vector_maker(varity_data.data, varity_data.qip_dict,args,weights.all_ones,False)
    print(weights.weights)
    #args_dict = {}
    #weights.fw_core_multiply_weight_vector_maker(varity_data.data,varity_data.qip_dict,args_dict)
   # print(args_dict)
    
