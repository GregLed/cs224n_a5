#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    
    def __init__(self, w_emb_sz):
        """Creates highway connection for NMT model

        Parameters
        ----------
        w_emb_sz : int
            word embedding size
        """
        super().__init__()
        self.proj = nn.Linear(in_features=w_emb_sz,out_features=w_emb_sz, bias=True)
        self.relu = nn.ReLU()
        self.gate = nn.Linear(in_features=w_emb_sz,out_features=w_emb_sz, bias=True)

    def forward(self,X_convout):
        """[summary]

        Parameters
        ----------
        X_convout : torch.tensor
            Convolution Network output tensor with size (b,e)
            where b = batch size, e = embedding size

        Returns
        -------
        torch.tensor
            size (b,e) where b = batch size, e = embedding size
        """

        X_proj = self.relu(self.proj(X_convout))
        X_gate = torch.sigmoid(self.gate(X_convout))

        X_hw = X_gate * X_proj + (1-X_gate) * X_convout
        
        return X_hw