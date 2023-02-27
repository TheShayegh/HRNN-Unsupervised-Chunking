# CUDA_VISIBLE_DEVICES=0 python3 test_script.py experiments/???/config.yml

import yaml
import sys
import pickle
import torch
from library.utils import data_padding, build_vocab
from library.HRNN import HRNNtagger, validate, eval_conll2000
from word_embeddings import get_embeddings
import os


def BIto21(heu_tag):
    num_tags = []
    for t in heu_tag[1:]:
        num_tags.append('1' if t=='I' else '2')
    num_tags.append('1')
    return num_tags


def main():
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
	
    if config['device'] != 'cpu' and not torch.cuda.is_available():
        raise Exception('THERE IS NO CUDA AVAILABLE!')
    else:
        device = torch.device(config['device'])
    print("device is:", device)

    data = pickle.load(open(config['test_data'], "rb"))
    original_tokens = [d[0] for d in data]
    word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = build_vocab(data)
    tokens, tags, msl = data_padding(data, word_to_ix, tag_to_ix, device=device)
    if config['load_last_test_embeddings'] and os.path.exists(config['home']+config['test_embeddings']):
        embeddings = torch.load(config['home']+config['test_embeddings'], map_location=device)
    else:
        config['embedding_path'] = config['test_embedding_path']
        embeddings = get_embeddings(tokens.to(device), ix_to_word, config, device)
        if config['test_embeddings']:
            torch.save(embeddings, config['home']+config['test_embeddings'])

    data = list(zip(embeddings, tags))

    hrnn_model = HRNNtagger(
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        tagset_size=2,
        device=device,
    ).to(device)
    hrnn_model.load_state_dict(torch.load(config['home']+config['best_model_path'], map_location=torch.device(device)))

    if config['test_mode'].lower() == 'enforced':
        enforced_tags = pickle.load(open(config['enforced_test_tags'], "rb"))
    else:
        enforced_tags = None

    loss, validation_output = validate(
        hrnn_model,
        data,
        [['B' for t in tags] for _,tags in data],
        device=device,
        enforced_tags=enforced_tags,
        enforced_mode=config['enforced_mode'].lower(),
    )
    flat_output = [line.split()[-1] for line in validation_output.strip().split('\n')]
    i = 0
    output = []
    for _, tags in data:
        output.append([])
        for j,t in enumerate(tags):
            if j:
                if not t:
                    break
                output[-1].append(flat_output[i])
            i += 1
    output = [BIto21(o) for o in output]
    output = list(zip(original_tokens, output))
    if 'target_path' in config:
        pickle.dump(output, open(config['target_path'], 'wb'))
    else:
        postfix = sys.argv[2]
        pickle.dump(output, open(config['test_data']+f'.{postfix}.pkl', 'wb'))

if __name__ == "__main__":
	main()
