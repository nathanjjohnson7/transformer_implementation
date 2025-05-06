import unicodedata
import string
import re


from PIL import Image
#import skimage.transform
import numpy as np
import random
import math
import glob

import io
import base64
#import cv2

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import xml.etree.ElementTree as ET
import os
from os import listdir
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import time

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
        encoded_pairs.append([[eng.word2index[x] for x in english_data] + [1], [0] + [fra.word2index[x] for x in french_data] + [1]])
        
    return pairs, encoded_pairs, eng, fra

def one_hot_encode_label(label, num_categories):
    final = T.zeros(label.shape[0], num_categories)
    new_label = T.cat((T.arange(label.shape[0]).unsqueeze(0), label.unsqueeze(0)))
    final[new_label[0], new_label[1]] = 1
    return final

#my implementation of the "Attention is All You Need" transform architecture starts here

class Positional_Encoding(nn.Module):
    def __init__(self, max_len=5000, d_model=512):
        super(Positional_Encoding, self).__init__()
        self.pe = T.zeros(max_len, d_model)
        self.pe[:, 0::2] = T.sin(T.arange(max_len).unsqueeze(1)/(10000**(T.arange(0, d_model, 2)/d_model)))
        self.pe[:, 1::2] = T.cos(T.arange(max_len).unsqueeze(1)/(10000**(T.arange(1, d_model, 2)/d_model)))
    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return x

#upper triangular mask that prevents a position from attending to future positions
def get_mask(seq_len):
    return T.triu(T.ones(seq_len, seq_len)* float('-inf'), diagonal=1)

#prevent the model from attending to padding tokens
def get_pad_mask(seq_len, pad_data):
    mask = T.ones(len(pad_data), seq_len)
    for i, num in enumerate(pad_data):
        if num == 0:
            continue
        mask[i, -num:] = 0
    return mask.unsqueeze(-1)

class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model, h=8):
        super(Multi_Head_Attention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.q_linear = nn.Linear(self.d_model, self.d_model*self.h)
        self.k_linear = nn.Linear(self.d_model, self.d_model*self.h)
        self.v_linear = nn.Linear(self.d_model, self.d_model*self.h)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.final_linear = nn.Linear(self.d_model*self.h, self.d_model)
    def forward(self, queries, keys, values, mask=None):
        #input.shape -> [batch_size, seq_len, d_model]
        batch_size = queries.shape[0]
        seq_len_q = queries.shape[1]
        seq_len_kv = keys.shape[1]
        
        queries = self.q_linear(queries)
        keys = self.k_linear(keys)
        values = self.v_linear(values)
        
        #reshape to [h, batch_size, seq_len_q, d_model]
        queries = T.permute(queries.view(batch_size, seq_len_q, self.d_model, self.h), (3, 0, 1, 2))
        #reshape to [h, batch_size, d_model, seq_len_kv]
        keys = T.permute(keys.view(batch_size, seq_len_kv, self.d_model, self.h), (3, 0, 2, 1))
        #reshape to [h, batch_size, seq_len_kv, d_model]
        values = T.permute(values.view(batch_size, seq_len_kv, self.d_model, self.h), (3, 0, 1, 2))
        
        multipled = T.matmul(queries, keys)
        scaled = multipled/math.sqrt(self.d_model)
        if mask!=None:
            scaled = scaled + mask
        softmaxed = self.softmax(scaled)
        
        final_matmul = T.matmul(softmaxed, values) #shape = [h, batch_size, seq_len_q, d_model]
        
        #new shape = [batch_size, seq_len_q, d_model*h]
        reshaped_final_matmul = T.permute(final_matmul, (1, 2, 3, 0)).reshape(batch_size, seq_len_q, self.d_model*self.h)
        
        output = self.final_linear(reshaped_final_matmul)
        
        return output

class Feed_Forward(nn.Module):
    def __init__(self, d_model):
        super(Feed_Forward, self).__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(self.d_model, self.d_model*4)
        self.linear2 = nn.Linear(self.d_model*4, self.d_model)
        self.relu = nn.ReLU(inplace = True)
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class Add_Norm(nn.Module):
    def __init__(self, d_model):
        super(Add_Norm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self, input1, input2):
        return self.layer_norm(input1+input2)

class Encoder_Module(nn.Module):
    def __init__(self, d_model=512):
        super(Encoder_Module, self).__init__()
        self.multi_head_attention = Multi_Head_Attention(d_model = 512)
        self.add_norm1 = Add_Norm(d_model)
        self.feed_forward = Feed_Forward(d_model)
        self.add_norm2 = Add_Norm(d_model)
        
    def forward(self, x):
        attended = self.multi_head_attention(x,x,x)
        add_normed = self.add_norm1(x, attended)
        feed_forwarded = self.feed_forward(add_normed)
        out = self.add_norm2(add_normed, feed_forwarded)
        return out

class Decoder_Module(nn.Module):
    def __init__(self, d_model = 512):
        super(Decoder_Module, self).__init__()
        self.masked_multi_head_attention = Multi_Head_Attention(d_model)
        self.add_norm1 = Add_Norm(d_model)
        self.multi_head_attention = Multi_Head_Attention(d_model)
        self.add_norm2 = Add_Norm(d_model)
        self.feed_forward = Feed_Forward(d_model)
        self.add_norm3 = Add_Norm(d_model)
        
    def forward(self, encoder_output, decoder_output, mask):
        mask_attended = self.masked_multi_head_attention(decoder_output, decoder_output, decoder_output, mask)
        add_normed1 = self.add_norm1(decoder_output, mask_attended)
        attended = self.multi_head_attention(mask_attended, encoder_output, encoder_output)
        add_normed2 = self.add_norm2(add_normed1, attended)
        feed_forwarded = self.feed_forward(add_normed2)
        out = self.add_norm3(add_normed2, feed_forwarded)
        return out


class Transformer(nn.Module):
    def __init__(self, num_input_embeddings, num_output_embeddings, d_model=512):
        super(Transformer, self).__init__()
        self.input_embedding = nn.Embedding(num_input_embeddings, d_model)
        self.output_embedding = nn.Embedding(num_output_embeddings, d_model)
        
        self.pos_encoding = Positional_Encoding()
        
        self.encoder1 = Encoder_Module()
        self.encoder2 = Encoder_Module()
        self.encoder3 = Encoder_Module()
        self.encoder4 = Encoder_Module()
        self.encoder5 = Encoder_Module()
        self.encoder6 = Encoder_Module()
        
        self.decoder1 = Decoder_Module()
        self.decoder2 = Decoder_Module()
        self.decoder3 = Decoder_Module()
        self.decoder4 = Decoder_Module()
        self.decoder5 = Decoder_Module()
        self.decoder6 = Decoder_Module()
        
        self.linear = nn.Linear(d_model, num_output_embeddings)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, encoder_input, decoder_input, encoder_pad_mask=None):
        assert encoder_input.shape[0] == decoder_input.shape[0]
        mask = get_mask(decoder_input.shape[1])
        
        encoder_input = self.input_embedding(encoder_input)
        
        encoder_input = self.pos_encoding(encoder_input)
        
        encoder_output = self.encoder1(encoder_input)
        encoder_output = self.encoder2(encoder_output)
        encoder_output = self.encoder3(encoder_output)
        encoder_output = self.encoder4(encoder_output)
        encoder_output = self.encoder5(encoder_output)
        encoder_output = self.encoder6(encoder_output)
        
        if encoder_pad_mask != None:
            encoder_output *= encoder_pad_mask
        
        decoder_input = self.output_embedding(decoder_input)
        
        decoder_input = self.pos_encoding(decoder_input)
        
        decoder_output = self.decoder1(encoder_output, decoder_input, mask)
        decoder_output = self.decoder2(encoder_output, decoder_output, mask)
        decoder_output = self.decoder3(encoder_output, decoder_output, mask)
        decoder_output = self.decoder4(encoder_output, decoder_output, mask)
        decoder_output = self.decoder5(encoder_output, decoder_output, mask)
        decoder_output = self.decoder6(encoder_output, decoder_output, mask)
        
        output = self.linear(decoder_output)
        #output = self.softmax(output)
        return output
    
    def run_encoder(self, encoder_input):
        encoder_input = self.input_embedding(encoder_input)

        encoder_input = self.pos_encoding(encoder_input)

        encoder_output = self.encoder1(encoder_input)
        encoder_output = self.encoder2(encoder_output)
        encoder_output = self.encoder3(encoder_output)
        encoder_output = self.encoder4(encoder_output)
        encoder_output = self.encoder5(encoder_output)
        encoder_output = self.encoder6(encoder_output)
        return encoder_output

    def run_decoder(self, encoder_output, decoder_input):
        mask = get_mask(decoder_input.shape[1])
        decoder_input = self.output_embedding(decoder_input)

        decoder_input = self.pos_encoding(decoder_input)

        decoder_output = self.decoder1(encoder_output, decoder_input, mask)
        decoder_output = self.decoder2(encoder_output, decoder_output, mask)
        decoder_output = self.decoder3(encoder_output, decoder_output, mask)
        decoder_output = self.decoder4(encoder_output, decoder_output, mask)
        decoder_output = self.decoder5(encoder_output, decoder_output, mask)
        decoder_output = self.decoder6(encoder_output, decoder_output, mask)

        output = self.linear(decoder_output)
        output = self.softmax(output)
        return output

    def evaluate(self, x, sos_index, eos_index = None, max_length = 5000):
        encoder_output = self.run_encoder(x)

        decoder_input = (T.ones([1])*sos_index).unsqueeze(1).int()
        while decoder_input.shape[1] < max_length or (eos_index != None and decoder_input[0][-1] != eos_index):
            out = self.run_decoder(encoder_output, decoder_input)
            decoder_input = T.cat((decoder_input, T.argmax(out, dim=-1)[0][-1].unsqueeze(0).unsqueeze(0)), dim=1)

        return decoder_input

def data_loader(dataset1, dataset2, batch_size):
    len_data = len(dataset1)
    indices = [i for i in range(len_data)]
    #indices = [14000 for i in range(len_data)]
    random.shuffle(indices)
    i = 0
    while True:
        if i < len_data//batch_size:
            real_batch_size = batch_size
        elif i == len_data//batch_size and len_data - (len_data//batch_size)*batch_size>0:
            real_batch_size = len_data - (len_data//batch_size)*batch_size
        else:
            i=0
            real_batch_size = batch_size
            indices = [i for i in range(len_data)]
            #indices = [14000 for i in range(len_data)]
            random.shuffle(indices)
            
        input1 = [dataset1[index][0] for index in indices[i*real_batch_size:(i+1)*real_batch_size]]
        max_size1 = max([len(x) for x in input1])
        in1_padding = [max_size1-len(array) for array in input1]
        input1 = T.tensor([array+([PAD_token]*(max_size1-len(array))) for array in input1])
        
        input2 = [dataset1[index][1] for index in indices[i*real_batch_size:(i+1)*real_batch_size]]
        max_size2 = max([len(x) for x in input2])
        in2_padding = [max_size2-len(array) for array in input2]
        input2 = T.tensor([array+([PAD_token]*(max_size2-len(array))) for array in input2])
        
        input3 = [dataset2[index][0] for index in indices[i*real_batch_size:(i+1)*real_batch_size]]
        input4 = [dataset2[index][1] for index in indices[i*real_batch_size:(i+1)*real_batch_size]]
        
        i+=1
        
        yield input1, input2, in1_padding, in2_padding, input3, input4

real_pairs, pairs, eng, fra = get_pairs_and_langs()
d_model = 512
h = 8

model = Transformer(eng.n_words, fra.n_words)
criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
lr = (d_model**-0.5)# * min((step_num**-0.5), (step_num*(warmup_steps**-1.5)))
#optimizer = T.optim.SGD(model.parameters(), lr=lr)
optimizer = T.optim.Adam(model.parameters(), lr = lr, betas = (0.9, 0.98), eps = 1e-09)
scheduler = T.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
epochs = 500
batch_size = 64
warmup_steps = 4000

my_dataloader = data_loader(pairs, real_pairs, batch_size)

losses = []
def train(epochs):
    model.train()
    for i in range(epochs):
        for g in optimizer.param_groups:
            g['lr'] = (d_model**-0.5) * min(((i+1e-5)**-0.5), (i*(warmup_steps**-1.5)))
        src, tgt, src_padding, tgt_padding, decoded_src, decoded_tgt = next(my_dataloader)
        encoder_pad_mask = get_pad_mask(src.shape[1], src_padding)
        decoder_pad_mask = get_pad_mask(tgt.shape[1]-1, tgt_padding)
        out = model(src, tgt[:,:-1], encoder_pad_mask)
        one_hot_tgt = T.stack([one_hot_encode_label(x, fra.n_words) for x in tgt[:,1:]])
        loss = criterion(out * decoder_pad_mask, one_hot_tgt * decoder_pad_mask)
        
        optimizer.zero_grad()
        loss.backward()
        #T.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        final_out = [fra.index2word[pred.item()] for pred in T.argmax(out[-1], dim=-1)]
        
        if len(losses) == 0 or min(losses) > loss.item():
            T.save(model.state_dict(), "model.pth")
            print("************* saved_model ******************** ", i)
        
        print("Epoch", i)
        print(loss.item())
        print("Input: {}".format(decoded_src[-1]))
        print()
        print("Target: {}".format(decoded_tgt[-1]))
        print()
        print("Output: {}".format(final_out))
        print()
        print()
        print()
        
        if i % 200 == 0 and i !=0:
            lr = scheduler.get_last_lr()[0]
            
        losses.append(loss.item())

train(epochs)
