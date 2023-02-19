# CUDA_VISIBLE_DEVICES=0 python3 train_script.py experiments/???/config.yml

import yaml
import sys
import pickle
import torch
from library.utils import data_padding, build_vocab
from library.HRNN import HRNNtagger, get_training_equipments, train, validate, eval_conll2000
from word_embeddings import get_embeddings
import numpy as np
import os


def _train(model, data, optimizer, scheduler, config, name, losses, device):
    loss = train(model, data, optimizer, scheduler, device=device)
    print( " __________________________________")
    print(f"| Train epoch {name}:")
    print(f"|     Loss:     {loss}")
    print( "|__________________________________")
    losses.append(loss)
    if config['train_loss']:
        with open(config['home']+config['train_loss'], 'wb') as f:
            pickle.dump(losses, f)
    return loss


def _validate(model, data, true_tags, config, name, losses, fscores, accs, device):
    if config['validation_mode'].lower() == 'enforced':
        enforced_tags = pickle.load(open(config['enforced_validation_tags'], "rb"))
    else:
        enforced_tags = None
    loss, validation_output = validate(
        model,
        data,
        true_tags,
        device=device,
        enforced_tags=enforced_tags,
        enforced_mode=config['enforced_mode'].lower(),
    )
    fscore, acc = eval_conll2000(validation_output)
    if config['validation_checkpoints_path']:
        pred_path = config['home']+config['validation_checkpoints_path'] + 'validation-' + str(name) + '.out'
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
    if config['validation_metrics']:
        plot_vars = {
            'loss': losses,
            'fscore': fscores,
            'acc': accs,
        }
        with open(config['home']+config['validation_metrics'], 'wb') as f:
            pickle.dump(plot_vars, f)
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
    
    if config['load_last_embeddings'] and os.path.exists(config['home']+config['validation_embeddings']):
        validation_embeddings = torch.load(config['home']+config['validation_embeddings'], map_location=device)
    else:
        config['embedding_path'] = config['val_embedding_path']
        validation_embeddings = get_embeddings(validation_tokens.to(device), ix_to_word, config, device)
        if config['validation_embeddings']:
            torch.save(validation_embeddings, config['home']+config['validation_embeddings'])
    
    if config['load_last_embeddings'] and os.path.exists(config['home']+config['train_embeddings']):
        training_embeddings = torch.load(config['home']+config['train_embeddings'], map_location=device)
    else:
        config['embedding_path'] = config['train_embedding_path']
        training_embeddings = get_embeddings(train_tokens.to(device), ix_to_word, config, device)
        if config['train_embeddings']:
            torch.save(training_embeddings, config['home']+config['train_embeddings'])

    training_data = list(zip(training_embeddings, train_tags))
    validation_data = list(zip(validation_embeddings, validation_tags))

    hrnn_model = HRNNtagger(
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        tagset_size=2,
        device=device,
    ).to(device)
    optimizer, scheduler = get_training_equipments(hrnn_model, lr=config['learning_rate'], num_iter=config['epocs'], warmup=config['warmup'])

    train_loss_vec = []
    validation_records = [], [], []
    validation_loss_vec, validation_fscore_vec, validation_acc_vec = validation_records
    best_fscore = 0.

    if config['pretrained_model']:
        hrnn_model.load_state_dict(torch.load(config['home']+config['pretrained_model'], map_location=torch.device(device)))
    _validate(
        hrnn_model,
        validation_data,
        validation_true_tags,
        config,
        'pre-trained' if config['pretrained_model'] else 'init model',
        *validation_records,
        device=device
    )

    for epoch in range(config['epocs']):
        print(f"============================ Epoch {epoch} ============================")
        print('LR:', scheduler.get_last_lr())
        _train(hrnn_model, training_data, optimizer, scheduler, config, epoch, train_loss_vec, device=device)
        fscore = _validate(hrnn_model, validation_data, validation_true_tags, config, epoch, *validation_records, device=device)

        if config['model_checkpoints_path'] and ( (epoch+1)%10 == 0 ):
            torch.save(hrnn_model.state_dict(), config['home']+config['model_checkpoints_path']+'hrnn'+str(epoch)+'.pt')
        if config['optimizer_path']:
            torch.save(optimizer.state_dict(), config['home']+config['optimizer_path'])

        if fscore > best_fscore:
            best_fscore = fscore
            torch.save(hrnn_model.state_dict(), config['home']+config['best_model_path'])


if __name__ == "__main__":
	main()