# Transformer Implementation

I implemented the "Attention is All You Need" paper which details the transformer architecture.

I have trained the model for English to French translation on a small subset of sentences, due to computational constraints. The dataset and data processing steps have been adopted from Pytorch's "NLP From Scratch: Translation with a Sequence to Sequence Network and Attention" tutorial. The tutorial implements a recurrent neural network, as opposed to our transformer implementation.

## Use

To train the model:

`$ python train.py`

Testing:

```bash
$ python test.py "They're amazing."

ils sont incroyables .
 
$ python test.py "I am happy."
 
je suis heureux .

$ python test.py "We are reliable."

nous sommes fiables .

$ python test.py "You are influential."

vous etes influent .
```



