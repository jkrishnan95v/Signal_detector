#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:53:34 2021

@author: jay
"""

"""
Cross Entropy 
"""


import torch
import torch.nn.functional as F
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss().cuda()

    def forward(self, logits, labels):
        
        loss = self.loss(logits, labels)
        return loss
