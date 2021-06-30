#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:25:33 2021

@author: jay
"""

import argparse
from utils.config import process_config

from agents import *

from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import torch


def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()

    # parse the config json file
    config = process_config(args.config)

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent]
    
    with h5py.File(config.data_root,'r') as hdf: #Read hdf5 file and converts into a numpy aray
            ls=list(hdf.keys())
            print('Dataset List: \n', ls)
            X =  np.array(hdf.get('extracted_x'))#extrated_x
            y = np.array(hdf.get('target_y'))#target_y
            X = torch.from_numpy(X)
            y = torch.from_numpy(y)
            #X = X.cuda()
            #y = y.cuda()
            
        
    config.X_train,config.X_test,config.y_train,config.y_test = train_test_split(X,y,test_size=0.090909,shuffle=False)
    config.X_train, config.X_val, config.y_train, config.y_val  = train_test_split(config.X_train, config.y_train, test_size=0.1, shuffle = False)
        #y_train = y_train.type(torch.LongTensor)
        #y_test = y_test.type(torch.LongTensor)
        #y_val = y_val.type(torch.LongTensor)
    config.num_rows, config.num_cols = config.X_train.shape
    config.num_rows_2, config.num_cols_2 = config.X_test.shape   
    config.num_rows_3, config.num_cols_3 = config.X_val.shape

    agent = agent_class(config)
    
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
