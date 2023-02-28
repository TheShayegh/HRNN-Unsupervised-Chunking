import torch
import numpy as np
from tqdm import tqdm
from transformers import *
from library.logger import timing_logger
from library.measure import Measure
from library.utils import select_indices, group_indices
import io


def get_embeddings(
    data_tokens: list[list],
    ix_to_word: dict,
    config: dict,
    device: torch.device,
) -> torch.Tensor:

    if config['embedding_mode'].lower()=='bert':
        return compute_emb_by_bert(data_tokens, ix_to_word, config=config, device=device)
    elif config['embedding_mode'].lower()=='file':
        return compute_emb_from_file(data_tokens, ix_to_word, config=config, device=device)
    elif config['embedding_mode'].lower()=='allones':
        return torch.ones([len(data_tokens), len(data_tokens[0]), 300]) 


@timing_logger
def compute_emb_from_file(
    data_tokens: list[list],
    ix_to_word: dict,
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    
    def load_vec(emb_path):
        vectors = []
        word2id = {}
        with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            next(f)
            for i, line in enumerate(f):
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                assert word not in word2id, 'word found twice'
                vectors.append(vect)
                word2id[word] = len(word2id)
        id2word = {v: k for k, v in word2id.items()}
        embeddings = np.vstack(vectors)
        return embeddings, id2word, word2id

    src_embeddings, src_id2word, src_word2id = load_vec(config['embedding_path'])
    feat_sents = []
    for idx, s_tokens in enumerate(tqdm(data_tokens)):
        feat_sents.append([src_embeddings[src_word2id.get(ix_to_word[ix], src_word2id['UNK'])].tolist() for ix in s_tokens.cpu().numpy()])
        # feat_sents.append([src_embeddings[src_word2id.get(ix_to_word[ix].lower(), src_word2id['unk'])].tolist() for ix in s_tokens.cpu().numpy()])
    feat_sents = torch.as_tensor(feat_sents).to(device)
    return feat_sents


@timing_logger
def compute_emb_by_bert(
    data_tokens: list[list],
    ix_to_word: dict,
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    token_heuristic = config['embedding_token_heuristic']

    scores = dict()
    syn_dists_all = dict()
    max_seq_len = 0
   
    model_class = AutoModel
    tokenizer_class = AutoTokenizer
    model_config = AutoConfig
    pretrained_weights = config['bert_pretrained_weights']

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, cache_dir='LM/cache')
    model = model_class.from_pretrained(pretrained_weights, cache_dir='LM/cache', 
        output_hidden_states=True, output_attentions=True).to(device)

    with torch.no_grad():
        test_sent = tokenizer.encode('test', add_special_tokens=False)
        token_ids = torch.tensor([test_sent]).to(device)
        all_hidden, all_att = model(token_ids)[-2:]
        
        n_layers = len(all_att)
        n_att = all_att[0].size(1)
        n_hidden = all_hidden[0].size(-1)
    
    measure = Measure(n_layers, n_att)
    feat_sents = torch.zeros([len(data_tokens), len(data_tokens[0]), config['embedding_dim']]) 

    for idx, s_tokens in enumerate(tqdm(data_tokens)):
        #################### read words and extract ##############
        s_tokens = [ix_to_word[ix] for ix in s_tokens.cpu().numpy()]

        raw_tokens = s_tokens
        s = ' '.join(s_tokens)
        tokens = tokenizer.tokenize(s)

        token_ids = tokenizer.encode(s, add_special_tokens=False)
        token_ids_tensor = torch.tensor([token_ids]).to(device)
        with torch.no_grad():
            all_hidden, all_att = model(token_ids_tensor)[-2:]
        all_hidden = list(all_hidden[1:])
        
        # (n_layers, seq_len, hidden_dim)
        all_hidden = torch.cat([all_hidden[n] for n in range(n_layers)], dim=0)
        
        #################### further pre processing ##############
        if len(tokens) > len(raw_tokens):
            th = token_heuristic
            if th == 'first' or th == 'last':
                mask = select_indices(tokens, raw_tokens, pretrained_weights, th)
                assert len(mask) == len(raw_tokens)
                all_hidden = all_hidden[:, mask]

            else:
                mask = group_indices(tokens, raw_tokens, pretrained_weights)
                raw_seq_len = len(raw_tokens)
                all_hidden = torch.stack(
                    [all_hidden[:, mask == i].mean(dim=1)
                     for i in range(raw_seq_len)], dim=1)
    
        all_hidden = all_hidden[n_layers - 1]
        feat_sents[idx] = all_hidden	
    
    return feat_sents