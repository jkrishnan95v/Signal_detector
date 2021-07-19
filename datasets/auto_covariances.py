#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:56:05 2021

@author: jay
"""

import numpy as np
import logging

import h5py
from sklearn.model_selection import train_test_split

import torch
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.data import DataLoader, TensorDataset, Dataset

class mydata(Dataset):
    def __init__(self, X, y, num_rows):
        self.raw_data = X
        self.labels = y
        self.rows = num_rows#why
        
    def __len__(self):
        return self.rows#oh thats why
    
    def __getitem__(self, idx):
        image = torch.tensor(self.raw_data[idx], dtype=torch.float)# / 0.15  # Normalize
        #image = image.view(2, 20, 20)  # (channel, height, width)
        label = self.labels[idx]
        return (image, label)
    
class covariance_dataloader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Array_DataLoader")
        
        if self.config.mode == "train":
            
            train_set = mydata(self.X_train, self.y_train, self.num_rows)
            valid_set = mydata(self.X_val, self.y_val, self.num_rows_3)
            
            
            self.logger.info("Loading DATA.....")
            
            self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=True)
           # self.train_iterations = (len(train_set) + self.config.batch_size) // self.config.batch_size
           # self.valid_iterations = (len(valid_set) + self.config.batch_size) // self.config.batch_size

            
            self.train_iterations = len(self.train_loader)
            self.valid_iterations = len(self.valid_loader)
        elif self.config.mode == "initial":
            
            pass
            
        elif self.config.mode == "test":
            
            test_set = mydata(self.X_test, self.y_test, self.num_rows_2)
            
            self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False)
           
            self.test_iterations = (len(test_set) + self.config.batch_size) // self.config.batch_size
            

         
        elif config.mode == "random":
            train_data = torch.randn(self.config.batch_size, self.config.input_channels, self.config.img_size, self.config.img_size)
            train_labels = torch.ones(self.config.batch_size).long()
            valid_data = train_data
            valid_labels = train_labels
            self.len_train_data = train_data.size()[0]
            self.len_valid_data = valid_data.size()[0]

            self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
            self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size

            train = TensorDataset(train_data, train_labels)
            valid = TensorDataset(valid_data, valid_labels)

            self.train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=False)

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def finalize(self):
        pass
