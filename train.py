import unicodedata
import string
import re

import random
import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
from os import listdir
from os.path import join

from transformer import Transformer
from utils import get_pad_mask, get_loss_mask, one_hot_encode_label

#the following function has been adapted from https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

#the following function has been adapted from https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

#the following class has been adapted from https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
SOS_token = 0
EOS_token = 1
PAD_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3  # Count SOS, EOS and PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

MAX_LENGTH = 5
#eng_prefixes, filterPair and filterPairs have been adopted from https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[0].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

#the following code has been adapted from https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def get_pairs_and_langs():
    lines = open("fra.txt", encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]
    
    eng = Lang("eng")
    fra = Lang("fra")
    
    pairs = filterPairs(pairs)
    
    for pair in pairs:
        eng.addSentence(pair[0])
        fra.addSentence(pair[1])
        
    encoded_pairs = []
    for pair in pairs:
        english_data = pair[0].split(' ')
        french_data = pair[1].split(' ')
        #add end-of-sentence token to English sentence after encoding
        #add start-of-sentence and end-of-sentence token to French sentence after encoding
        encoded_pairs.append([[eng.word2index[x] for x in english_data] + [1], [0] + [fra.word2index[x] for x in french_data] + [1]])
        
    return pairs, encoded_pairs, eng, fra

def data_loader(dataset1, dataset2, batch_size):
    #dataset1 and dataset2 are both lists of English-French sentence pairs
    #in dataset1, the data is represented as integers
    #in dataset2, the data is in the form of raw text


    len_data = len(dataset1)
    indices = [i for i in range(len_data)]
    #indices = [14000 for i in range(len_data)]
    random.shuffle(indices)
    i = 0
    while True:
        if i < len_data//batch_size:
            #if i is less than the integer part of len_data/batch_size, we just use the batch_size
            real_batch_size = batch_size
        elif i == len_data//batch_size and len_data - (len_data//batch_size)*batch_size>0:
            #if we don't have a whole batch_left but a few remaining, the batch_size equals the remanant
            real_batch_size = len_data - (len_data//batch_size)*batch_size
        else:
            #at this point we've made a complete loop over all the data and we restart the generator
            # by creating new random indicies
            i=0
            real_batch_size = batch_size
            indices = [i for i in range(len_data)]
            #indices = [14000 for i in range(len_data)]
            random.shuffle(indices)
        
        #get current batch of encoded english sentences and pad it. Store the padding amount
        input1 = [dataset1[index][0] for index in indices[i*real_batch_size:(i+1)*real_batch_size]]
        #we pad the whole batch to the length of the longest sentence
        max_size1 = max([len(x) for x in input1])
        in1_padding = [max_size1-len(array) for array in input1]
        input1 = T.tensor([array+([PAD_token]*(max_size1-len(array))) for array in input1])

        #get current batch of encoded French sentences and pad it. Store the padding amount
        input2 = [dataset1[index][1] for index in indices[i*real_batch_size:(i+1)*real_batch_size]]
        max_size2 = max([len(x) for x in input2])
        in2_padding = [max_size2-len(array) for array in input2]
        input2 = T.tensor([array+([PAD_token]*(max_size2-len(array))) for array in input2])
        
        #get current batch of raw English and French sentenches
        input3 = [dataset2[index][0] for index in indices[i*real_batch_size:(i+1)*real_batch_size]]
        input4 = [dataset2[index][1] for index in indices[i*real_batch_size:(i+1)*real_batch_size]]
        
        i+=1
        
        yield input1, input2, in1_padding, in2_padding, input3, input4

def train(epochs):
    real_pairs, pairs, eng, fra = get_pairs_and_langs()
    d_model = 512
    h = 8

    model = Transformer(eng.n_words, fra.n_words)
    criterion = nn.CrossEntropyLoss()
    lr = (d_model**-0.5)
    optimizer = T.optim.Adam(model.parameters(), lr = lr, betas = (0.9, 0.98), eps = 1e-09)
    batch_size = 64
    warmup_steps = 4000

    my_dataloader = data_loader(pairs, real_pairs, batch_size)

    model.train()
    losses = []
    for i in range(epochs):
        for g in optimizer.param_groups:
            if i == 0:
                g['lr'] = 0.0
            else:
                g['lr'] = (d_model ** -0.5) * min(i ** -0.5, i * (warmup_steps ** -1.5))
        src, tgt, src_padding, tgt_padding, decoded_src, decoded_tgt = next(my_dataloader)
        encoder_pad_mask = get_pad_mask(src.shape[1], src_padding)
        #we subtract 1 from the tgt.shape since we want the model to predict the next token
        # so there's no need to provide the last token. There isn't a next token to predict at the last token.
        decoder_pad_mask = get_pad_mask(tgt.shape[1]-1, tgt_padding)
        loss_mask = get_loss_mask(tgt.shape[1]-1, tgt_padding)
        out = model(src, tgt[:,:-1], encoder_pad_mask, decoder_pad_mask)
        #these are the next_token target prediction targets, starting from the first token, so we exclude the first token
        one_hot_tgt = T.stack([one_hot_encode_label(x, fra.n_words) for x in tgt[:,1:]])
        loss = criterion(out * loss_mask, one_hot_tgt * loss_mask)
        
        optimizer.zero_grad()
        loss.backward()
        #T.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        #let's take a look at the output of the last sample in the batch just to gauge performance
        final_out = [fra.index2word[pred.item()] for pred in T.argmax(out[-1], dim=-1)]
        
        if len(losses) == 0 or min(losses) > loss.item():
            T.save(model.state_dict(), "model.pth")
            print("************* saved_model ******************** ", i)

        print("Epoch", i)
        print("loss: ", loss.item())
        with open("log.txt", "a") as log_file:
            print("Epoch", i, file=log_file)
            print("loss: ", loss.item(), file=log_file)
            print("Input: {}".format(decoded_src[-1]), file=log_file)
            print(file=log_file)
            print("Target: {}".format(decoded_tgt[-1]), file=log_file)
            print(file=log_file)
            print("Output: {}".format(final_out), file=log_file)
            print("\n\n", file=log_file)

        with open("losses.txt", "a") as log_file:
            print(i, loss.item(), file=log_file)
        losses.append(loss.item())
        


if __name__ == "__main__":
    epochs = 500
    train(epochs)
