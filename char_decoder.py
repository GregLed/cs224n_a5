#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ch_dec_emb = self.decoderCharEmb(input) #Tensor: (length, batch_size, char_embed_size)
        ch_dec_hiddens, dec_hidden = self.charDecoder(ch_dec_emb, dec_hidden) #dec_hidden: (last_hidden, last_cell) 
        scores = self.char_output_projection(ch_dec_hiddens)

        return scores, dec_hidden

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        logits, dec_hidden = self.forward(char_sequence[:-1, ...], dec_hidden) #Tensor: (length, batch_size, num_of_classes->characters)
        logits  = logits.permute(1,2,0)
        target = char_sequence[1:, ...].permute(1,0) # we compare vs next character

        # cross entropy takes N,C,d1,d2,... order
        loss = F.cross_entropy(logits, 
                               target,
                               ignore_index=self.target_vocab.char_pad, 
                               reduction='sum') 

        return loss

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        ### END YOUR CODE

        dec_hidden = initialStates
        batch_size = dec_hidden[0].size(1)
        chars_ints = torch.zeros(batch_size, dtype=torch.long, device=device) + self.target_vocab.start_of_word
        results = []

        for i in range(max_length):
            logits, dec_hidden = self.forward(chars_ints.unsqueeze(0), dec_hidden) #Tensor: (length=1, batch_size, num_of_classes-> # of characters)
            logits = logits.squeeze(0) #Tensor: (batch_size, num_of_classes-> # of characters)
            chars_ints = torch.argmax(logits, dim=1) #Tensor: (batch_size)
            results.append(chars_ints)
        
        # convert to one tensor
        results = torch.stack(results)
        
        # transpose so that one work is in a row
        results = results.t().contiguous()

        # find when <end> token happens
        stop = results == self.target_vocab.end_of_word

        # convert to list for iteration
        results = results.tolist()

        decodedWords = []
        for i_b, batch in enumerate(results):
            batch_chars = []

            # iterate over chars and check if <end> token is present
            for i_ch, ch in enumerate(batch):
                if stop[i_b, i_ch]:
                    break
                batch_chars.append(ch)

            # convert to ints to chars
            decodedWords.append([self.target_vocab.id2char[ch] for ch in batch_chars])

        # join chars into one word
        decodedWords = [''.join(word) for word in decodedWords]

        return decodedWords