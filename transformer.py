import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import get_causal_mask

#implements equations found in section 3.5 of the "Attention is All You Need" paper
class Positional_Encoding(nn.Module):
    def __init__(self, max_len=5000, d_model=512):
        super(Positional_Encoding, self).__init__()
        self.pe = T.zeros(max_len, d_model)
        self.pe[:, 0::2] = T.sin(T.arange(max_len).unsqueeze(1)/
                                (10000**(T.arange(0, d_model, 2)/d_model)))
        self.pe[:, 1::2] = T.cos(T.arange(max_len).unsqueeze(1)/
                                (10000**(T.arange(1, d_model, 2)/d_model)))
    def forward(self, x):
        #x.shape -> [batch_size, seq_len, d_model=512]
        x = x + self.pe[:x.size(1)]
        return x

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
        
        multiplied = T.matmul(queries, keys)
        scaled = multiplied/math.sqrt(self.d_model)
        if mask!=None:
            scaled = scaled + mask
        softmaxed = self.softmax(scaled) #this tell us how much attention each query pays to each key
        
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
        
    def forward(self, x, encoder_pad_mask=None):
        attended = self.multi_head_attention(x,x,x, encoder_pad_mask)
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
        
    def forward(self, encoder_output, decoder_output, decoder_mask, encoder_pad_mask=None):
        mask_attended = self.masked_multi_head_attention(decoder_output, decoder_output, decoder_output, decoder_mask)
        add_normed1 = self.add_norm1(decoder_output, mask_attended)
        attended = self.multi_head_attention(mask_attended, encoder_output, encoder_output, encoder_pad_mask)
        add_normed2 = self.add_norm2(add_normed1, attended)
        feed_forwarded = self.feed_forward(add_normed2)
        out = self.add_norm3(add_normed2, feed_forwarded)
        return out

class Transformer(nn.Module):
    def __init__(self, num_input_embeddings, num_output_embeddings, d_model=512, num_stack=6):
        super(Transformer, self).__init__()
        self.input_embedding = nn.Embedding(num_input_embeddings, d_model)
        self.output_embedding = nn.Embedding(num_output_embeddings, d_model)
        
        self.pos_encoding = Positional_Encoding()

        self.num_stack = num_stack

        self.encoders = nn.ModuleList([Encoder_Module() for i in range(self.num_stack)])
        self.decoders = nn.ModuleList([Decoder_Module() for i in range(self.num_stack)])
        
        self.linear = nn.Linear(d_model, num_output_embeddings)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, encoder_input, decoder_input, encoder_pad_mask=None, decoder_pad_mask=None):
        assert encoder_input.shape[0] == decoder_input.shape[0] #batch_size must be equal
        
        encoder_input = self.input_embedding(encoder_input) #embed
        
        encoder_output = self.pos_encoding(encoder_input) #positionally encode

        for encoder in self.encoders:
            encoder_output = encoder(encoder_output, encoder_pad_mask)
        
        decoder_input = self.output_embedding(decoder_input)
        
        decoder_output = self.pos_encoding(decoder_input)

        #prevent decoder from attending to future tokens
        causal_mask = get_causal_mask(decoder_input.shape[1])
        #also prevent the decoder from attending to padding tokens
        decoder_mask = causal_mask if decoder_pad_mask == None else decoder_pad_mask + causal_mask

        for decoder in self.decoders:
            decoder_output = decoder(encoder_output, decoder_output, decoder_mask, encoder_pad_mask)
        
        output = self.linear(decoder_output)
        return output
    
    #run the encoder on one input sample
    def run_encoder(self, encoder_input):
        encoder_input = self.input_embedding(encoder_input)

        encoder_output = self.pos_encoding(encoder_input)

        for encoder in self.encoders:
            encoder_output = encoder(encoder_output)

        return encoder_output

    #run the decoder for one input sample
    def run_decoder(self, encoder_output, decoder_input):
        mask = get_causal_mask(decoder_input.shape[1])
        decoder_input = self.output_embedding(decoder_input)

        decoder_output = self.pos_encoding(decoder_input)

        for decoder in self.decoders:
            decoder_output = decoder(encoder_output, decoder_output, mask)

        output = self.linear(decoder_output)
        output = self.softmax(output)
        return output

    #given the encoder output of one sample, this function generates text, token by token
    def evaluate(self, x, sos_index, eos_index = None, max_length = 5000):
        encoder_output = self.run_encoder(x)

        decoder_output = (T.ones([1])*sos_index).unsqueeze(1).int() #provide Start-of_sentence token
        #generate tokens, one by one until we encounter an end-of-sentence token or we exceed max_length
        while (eos_index != None and decoder_output[0][-1].item() != eos_index) and decoder_output.shape[1] < max_length:
            #generate token and concatenate to output
            out = self.run_decoder(encoder_output, decoder_output)
            decoder_output = T.cat((decoder_output, T.argmax(out, dim=-1)[0][-1].unsqueeze(0).unsqueeze(0)), dim=1)

        return decoder_output
