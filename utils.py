import torch as T

def one_hot_encode_label(label, num_categories):
    final = T.zeros(label.shape[0], num_categories)
    new_label = T.cat((T.arange(label.shape[0]).unsqueeze(0), label.unsqueeze(0)))
    final[new_label[0], new_label[1]] = 1
    return final

#this prevents the decoder from attending to future tokens
def get_causal_mask(seq_len):
    return T.triu(T.ones(seq_len, seq_len)* float('-inf'), diagonal=1)

#mask out padding
def get_pad_mask(seq_len, pad_data):
    #pad_data is a list of length batch_size, and the i-th value denotes how many padding token
    # have been added to the i-th sequence in the batch
    batch_size = len(pad_data)
    mask = T.zeros((batch_size, 1, seq_len))
    column_idx = T.arange(seq_len).view(1,1,seq_len)
    #how many tokens in each sequence are not padding
    remain = (seq_len - T.tensor(pad_data)).view(batch_size,1,1)
    mask[column_idx>=remain] = float('-inf')#-1e9
    return mask #[batch_size, 1, seq_len]

#mask out padded tokens for loss calculations
def get_loss_mask(seq_len, pad_data):
    #pad_data is a list of length batch_size, and the i-th value denotes how many padding token
    # have been added to the i-th sequence in the batch
    batch_size = len(pad_data)
    mask = T.zeros((batch_size, seq_len, 1))
    row_idx = T.arange(seq_len).view(1,seq_len,1)
    #how many tokens in each sequence are not padding
    remain = (seq_len - T.tensor(pad_data)).view(batch_size,1,1)
    mask[row_idx<remain] = 1
    return mask #[batch_size, seq_len, 1]
