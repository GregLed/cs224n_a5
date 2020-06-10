#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    
    kernel_sz = 5
    padding_sz = 1
    
    def __init__(self, ch_emb_sz, w_emb_sz, max_w_len):
        """[summary]

        Parameters
        ----------
        ch_emb_sz : int
            character embedding dimension
        w_emb_sz : int
            word embedding dimension (number of filter in convolution)
        max_w_len : int
            maximum word length (IS IT NEEDED?)
        """

        super().__init__()
        self.ch_emb_sz = ch_emb_sz
        self.w_emb_sz = w_emb_sz
        self.max_w_len = max_w_len
        self.conv = nn.Conv1d(in_channels=self.ch_emb_sz, out_channels=self.w_emb_sz, 
                        kernel_size=self.kernel_sz, stride=1, padding=self.padding_sz)
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
    
    def forward(self, X_reshaped):
        """[summary]

        Parameters
        ----------
        X_reshaped : tensor
            char embeddings tensor with size (b, c_e, m)
            where b = batch size, c_e = char embedding size, m - max word length

        Returns
        -------
        tensor
            final output of the Convolution Network with size (b, w_e)
            where b = batch size, w_e = word embedding size
            conv -> relu -> maxpool -- one number per filter
        """

        X_conv = self.conv(X_reshaped)
        X_convout = self.maxpool(self.relu(X_conv))
        X_convout = X_convout.squeeze(-1)

        return X_convout