'''implements class and methods to load and verify data for varity models
    a json file will provide parameters necessary for correct data processing'''

import pandas as pd
import numpy as np
import json


class Dataloader_Varity():
    '''load training data for varity, verify that quality informative properties '''

    def __init__(self,json_file_path: str) -> None:

        self.data_path: str = None
        self.config_dict: dict = None
        self.qips: set = set()
        self.feature_set: set = set()
        self._load_csv(json_file_path)

    def _load_csv(self,json_file_path: str) -> None:
        self.config_dict = self._load_config(json_file_path)
        self.data_path = self.config_dict["train_data_path"]
        self.qip_dict = self.config_dict["qip_dict"]
        self.feature_set = set(self.config_dict["list_features"])
        #check for valid path
        if self.data_path == "":
            raise ValueError("empty training data path provided, please provide a valid path")

        #read in data columns (first row of CSV) for comparisions later on
        data = pd.read_csv(self.data_path,low_memory=False,index_col=False,nrows=1)
        data_fields = set(data.columns)

        #set the qip values, by iterating through the qip_dict
        for data_group in self.qip_dict:
            for data_subset in self.qip_dict[data_group]:
                for qip in self.qip_dict[data_group][data_subset]:
                    self.qips.add(qip)


        #validate config file, by comparing features and qips with columns provided in training data 
        if self.feature_set.issubset(data_fields) and self.qips.issubset(data_fields):
            data = pd.read_csv(self.data_path,low_memory=False,index_col=False)
            self.data = data
        elif (self.feature_set-data_fields):
            raise ValueError(f"feature(s) in configuration file not present in data provided - {self.feature_set-data_fields} missing")

        elif (self.qips-data_fields):#a-b -> items present in a that are not in b
            raise ValueError(f"QIP in configuration file not present in data provided - {self.qips-data_fields} missing")



    def _load_config(self,json_file_path: str) -> dict:
        with open(json_file_path,'r') as f:
            config_dict = json.load(f)
        return config_dict

if __name__ == "__main__":
    a = Dataloader_Varity("/Users/alirezarasoulzadeh/Desktop/reimplemented_varity/test_config.json")
    print(a.data)
