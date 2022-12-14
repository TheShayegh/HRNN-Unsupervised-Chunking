import yaml
import sys
import pickle
import torch
from library.utils import data_padding, build_vocab
from library.HRNN import HRNNtagger, validate, eval_conll2000
from word_embeddings import get_embeddings


def _test(model, data, true_tags, config, device):
    loss, validation_output = validate(model, data, true_tags, device=device)
    fscore, acc = eval_conll2000(validation_output)
    if config['test_output_path']:
        with open(config['test_output_path'], 'w') as f:
            f.write(validation_output)
    print( " __________________________________")
    print(f"| Test:")
    print(f"|     Loss:     {loss}")
    print(f"|     F1:       {fscore}")
    print(f"|     Accuracy: {acc}")
    print( "|__________________________________")
    return fscore, loss, acc


def main():
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
	
    if config['device'] != 'cpu' and not torch.cuda.is_available():
        raise Exception('THERE IS NO CUDA AVAILABLE!')
    else:
        device = torch.device(config['device'])
    print("device is:", device)

    test_data = pickle.load(open(config['test_data'], "rb"))
    test_true_tags = pickle.load(open(config['test_true_tags'], "rb"))
    word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = build_vocab(test_data)
    test_tokens, test_tags, test_msl = data_padding(test_data, word_to_ix, tag_to_ix, device=device)
    if config['load_last_test_embeddings']:
        test_embeddings = torch.load(config['test_embeddings'], map_location=device)
    else:
        test_embeddings = get_embeddings(test_tokens.to(device), ix_to_word, config, device)
        if config['test_embeddings']:
            torch.save(test_embeddings, config['test_embeddings'])

    test_data = list(zip(test_embeddings, test_tags))

    hrnn_model = HRNNtagger(
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        tagset_size=len(tag_to_ix),
        device=device,
    ).to(device)
    hrnn_model.load_state_dict(torch.load(config['best_model_path'], map_location=torch.device(device)))
    fscore, loss, acc = _test(hrnn_model, test_data, test_true_tags, config, device) 


if __name__ == "__main__":
	main()
