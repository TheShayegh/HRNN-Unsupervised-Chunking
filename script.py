import yaml
import sys
import pickle
import torch
# from transformers import *
from library.utils import data_padding, build_vocab
from library.HRNN import HRNNtagger, get_training_equipments, train, validate, eval_conll2000
from word_embeddings import get_embeddings
import numpy as np


def _train(model, data, optimizer, name, losses, device):
    loss = train(model, data, optimizer, device=device)
    print( " __________________________________")
    print(f"| Train epoch {name}:")
    print(f"|     Loss:     {loss}")
    print( "|__________________________________")
    losses.append(loss)


def _validate(model, data, true_tags, config, name, losses, fscores, accs, device):
    loss, validation_output = validate(model, data, device=device)
    fscore, acc = eval_conll2000(validation_output)
    if config['validation_checkpoints_path']:
        pred_path = config['validation_checkpoints_path'] + 'validation-' + str(name) + '.out'
        with open(pred_path, 'w') as f:
            f.write(validation_output)
    print( " __________________________________")
    print(f"| Validation {name}:")
    print(f"|     Loss:     {loss}")
    print(f"|     F1:       {fscore}")
    print(f"|     Accuracy: {acc}")
    print( "|__________________________________")
    losses.append(loss)
    fscores.append(fscore)
    accs.append(acc)
    return fscore
    

def main():
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
	
    if config['device'] != 'cpu' and not torch.cuda.is_available():
        raise Exception('THERE IS NO CUDA AVAILABLE!')
    else:
        device = torch.device(config['device'])
    print("device is:", device)

    training_data = pickle.load(open(config['train_data'], "rb"))
    validation_data = pickle.load(open(config['validation_data'], "rb"))
    validation_true_tags = pickle.load(open(config['validation_true_tags'], "rb"))
    word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = build_vocab(training_data + validation_data)
    train_tokens, train_tags, train_msl = data_padding(training_data, word_to_ix, tag_to_ix, device=device)
    validation_tokens, validation_tags, validation_msl = data_padding(validation_data, word_to_ix, tag_to_ix, device=device)
    if config['load_last_embeddings']:
        validation_embeddings = torch.load(config['validation_embeddings'], map_location=device)
        training_embeddings = torch.load(config['train_embeddings'], map_location=device)
    else:
        validation_embeddings = get_embeddings(validation_tokens.to(device), ix_to_word, config, device)
        torch.save(validation_embeddings, config['validation_embeddings'])
        training_embeddings = get_embeddings(train_tokens.to(device), ix_to_word, config, device)
        torch.save(training_embeddings, config['train_embeddings'])

    training_data = list(zip(training_embeddings, train_tags))
    validation_data = list(zip(validation_embeddings, validation_tags))

    hrnn_model = HRNNtagger(
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        tagset_size=len(tag_to_ix),
        device=device,
    ).to(device)
    optimizer, scheduler = get_training_equipments(hrnn_model, lr=config['learning_rate'], num_iter=config['epocs'], warmup=config['warmup'])

    train_loss_vec = []
    validation_records = [], [], []
    validation_loss_vec, validation_fscore_vec, validation_acc_vec = validation_records
    best_fscore = 0.

    if config['pretrained_model']:
        hrnn_model.load_state_dict(torch.load(config['pretrained_model'], map_location=torch.device(device)))
        _validate(hrnn_model, validation_data, validation_true_tags, config, 'pre-trained', *validation_records, device=device)

    for epoch in range(config['epocs']):
        print(f"============================ Epoch {epoch} ============================")
        _train(hrnn_model, training_data, optimizer, epoch, train_loss_vec, device=device)
        fscore = _validate(hrnn_model, validation_data, validation_true_tags, config, epoch, *validation_records, device=device)
        scheduler.step()
        print('LR:', scheduler.get_last_lr())

        if fscore > best_fscore:
            best_fscore = fscore
            torch.save(hrnn_model.state_dict(), config['best_model_path'])	

        if config['model_checkpoints_path'] and ( (epoch+1)%10 == 0 ):
            torch.save(hrnn_model.state_dict(), config['model_checkpoints_path']+'hrnn'+str(epoch)+'.pt')

        if config['optimizer_path']:
            torch.save(optimizer.state_dict(), config['optimizer_path'])

        if config['plots_dir']:
            plot_vars = {
                'train_loss_vec': train_loss_vec,
                'validation_loss_vec': validation_loss_vec,
                'validation_fscore_vec': validation_fscore_vec,
                'validation_acc_vec': validation_acc_vec,
            }
            with open(config['plots_dir']+'plot-data.pkl', 'wb') as f:
                pickle.dump(plot_vars, f)


if __name__ == "__main__":
	main()