from train import get_pairs_and_langs, normalizeString
from transformer import Transformer
import torch as T
import argparse

_, _, eng_lang, fra_lang = get_pairs_and_langs()

model = Transformer(eng_lang.n_words, fra_lang.n_words)
model.load_state_dict(T.load("model.pth"))


parser = argparse.ArgumentParser(description="Translate English to French using Transformer.")
parser.add_argument("sentence", type=str, help="Input English sentence to translate")

args = parser.parse_args()

sentence = normalizeString(args.sentence)
encoded = [eng_lang.word2index[x] for x in sentence.lower().split(" ")]

output = model.evaluate(T.tensor([encoded]), sos_index=0, eos_index=1)

print(" ".join([fra_lang.index2word[i.item()] for i in output[0][1:-1]]))
