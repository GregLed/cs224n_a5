#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        self.word_embed_size = word_embed_size
        self.char_embed_size = 50
        self.max_word_len = 21
        self.dropout_p = 0.3

        self.ch_emb = nn.Embedding(len(vocab.char2id), self.char_embed_size, padding_idx=vocab.char_pad)
        self.cnn = CNN(self.char_embed_size, self.word_embed_size, self.max_word_len)
        self.hw = Highway(self.word_embed_size)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """

        X_emb = self.ch_emb(input) # Tensor: (max_sentence_length, batch_size, max_word_length, char_embed_size)
        X_emb_reshaped = X_emb.reshape(X_emb.size(0) * X_emb.size(1), X_emb.size(2), X_emb.size(3)) # Tensor: (max_sentence_length*batch_size, max_word_length, char_embed_size)
        X_emb_reshaped = X_emb_reshaped.permute(0,2,1) # Tensor: (max_sentence_length*batch_size, char_embed_size, max_word_length)
        X_convout = self.cnn(X_emb_reshaped)
        X_hw = self.hw(X_convout)
        output = self.dropout(X_hw)
        output = output.reshape(input.size(0), input.size(1), X_convout.size(1))

        return output