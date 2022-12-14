import gensim
import random
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from subprocess import run, PIPE
from library.logger import timing_logger
# from transformers import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


@timing_logger
def build_vocab(data):
	word_to_ix = {"PAD": 0, "UNK": 1, "SOS": 2, "EOS": 3}
	net = [d for d,_ in data]
	
	for sent in net:
		for word in sent:
			if word not in word_to_ix:
				word_to_ix[word] = len(word_to_ix)
	
	ix_to_word = {v: k for k, v in word_to_ix.items()}
	tag_to_ix = {"<pad>": 0, "1": 1, "2": 2, "3": 3}
	ix_to_tag = {v: k for k, v in tag_to_ix.items()}

	return word_to_ix, ix_to_word, tag_to_ix, ix_to_tag


def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] for w in seq]
	return torch.tensor(idxs, dtype=torch.long)


@timing_logger
def data_padding(data, word_to_ix, tag_to_ix, device, max_seq_len=20):
	data_lengths = [len(sentence)+2 for sentence,tags in data]
	
	max_seq_len = max(data_lengths)
	padded_data = torch.empty(len(data), max_seq_len, dtype=torch.long).to(device)
	padded_data.fill_(0.)
	# copy over the actual sequences
	for i, x_len in enumerate(data_lengths): 
		sequence,tags = data[i]
		sequence.insert(0,'SOS')
		sequence.append('EOS')
		
		sequence = prepare_sequence(sequence, word_to_ix)
		padded_data[i, 0:x_len] = sequence[0:x_len]

	tag_lengths = [len(tags)+2 for sentence, tags in data]
	padded_tags = torch.empty(len(data), max_seq_len, dtype=torch.long).to(device)
	padded_tags.fill_(0.)

	for i, y_len in enumerate(tag_lengths):
		sequence,tags = data[i]
		tags.insert(0,'<pad>')  # for SOS
		tags.append('<pad>')  ## for EOS

		tags = prepare_sequence(tags, tag_to_ix)
		padded_tags[i, 0:y_len] = tags[:y_len]

	return padded_data, padded_tags, max_seq_len


specials = {'bert': '#', 'gpt2': 'Ġ', 'xlnet': '▁', 'roberta': 'Ġ'}


def select_indices(tokens, raw_tokens, model, mode):
    mask = []
    raw_i = 0
    collapsed = ''
    model = model.split('-')[0]
    special = specials[model]

    for i in range(len(tokens)):
        token = tokens[i]

        while len(token) > 0 and token[0] == special:
            token = token[1:]
        if collapsed == '' and len(token) > 0:
            start_idx = i
        collapsed += token
        if collapsed == raw_tokens[raw_i]:
            if mode == 'first':
                mask.append(start_idx)
            elif mode == 'last':
                mask.append(i)
            else:
                raise NotImplementedError
            raw_i += 1
            collapsed = ''
    if raw_i != len(raw_tokens):
        raise Exception(f'Token mismatch: \n{tokens}\n{raw_tokens}')
    return mask


def group_indices(tokens, raw_tokens, model):
    mask = []
    raw_i = 0
    collapsed = ''
    model = model.split('-')[0]
    special = specials[model]

    for i in range(len(tokens)):
        token = tokens[i]

        while len(token) > 0 and token[0] == special:
            token = token[1:]
        collapsed += token
        mask.append(raw_i)
        if collapsed == raw_tokens[raw_i]:
            raw_i += 1
            collapsed = ''
    if raw_i != len(raw_tokens):
        return 
    return torch.tensor(mask)