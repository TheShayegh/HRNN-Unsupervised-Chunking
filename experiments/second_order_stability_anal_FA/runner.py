import yaml
import sys
import os
from copy import deepcopy

def main():
    print(f"*** CUDA_VISIBLE_DEVICES = {sys.argv[2]}\n")
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    home = config['home']

    for i in range(10):
        print(f"*** i = {i}")
        config_i = deepcopy(config)
        config_i['best_model_path'] = config_i['source_model_path'].format(i)

        for data in ['train', 'validation', 'test']:
            config_idata = deepcopy(config_i)
            config_idata['test_data'] = config[data+'_data']
            config_idata['test_embeddings'] = 'source_'+config[data+'_embeddings']
            config_idata['target_path'] = config['target_path'].format(home, data, i)

            with open(config['source_config']) as f:
                source_config = yaml.load(f, Loader=yaml.FullLoader)
                
                for item in ['device', 'embedding_dim', 'hidden_dim', 'learning_rate', 'warmup', 'epocs',
                    'embedding_mode', 'train_embedding_path', 'val_embedding_path', 'test_embedding_path',
                    'embedding_token_heuristic', 'bert_pretrained_weights', 'validation_mode', 'test_mode',
                    'enforced_mode']:
                    config_idata[item] = source_config[item]

            if os.path.exists('../../'+config_idata['target_path']):
                continue
        
            with open("_config.yml", 'w') as f:
                f.write(yaml.dump(config_idata))

            command = f'cd ../../ && CUDA_VISIBLE_DEVICES={sys.argv[2]} python3 use_script.py {home}_config.yml'
            print("*****", command)
            x = os.system(command)
            if x:
                print(x)
                exit(x)


        for j in range(10):
            print(f"*** j = {j}")
            config_ij = deepcopy(config)
            config_ij['test_output_path'] = config['test_output_path'].format(i, j)

            if os.path.exists(config_ij['test_output_path']):
                continue

            # config_ij['train_loss'] = config['train_loss'].format(i, j)
            # config_ij['validation_metrics'] = config['validation_metrics'].format(i, j)
            config_ij['best_model_path'] = config['best_model_path'].format(i, j)
            for data in ['train', 'validation', 'test']:
                config_ij[data+'_data'] = config['target_path'].format(home, data, i)

            with open("_config.yml", 'w') as f:
                f.write(yaml.dump(config_ij))

            command = f'cd ../../ && CUDA_VISIBLE_DEVICES={sys.argv[2]} python3 train_script.py {home}_config.yml'
            print("*****", command)
            x = os.system(command)
            if x:
                print(x)
                exit(x)

            command = f'cd ../../ && CUDA_VISIBLE_DEVICES={sys.argv[2]} python3 test_script.py {home}_config.yml'
            print("*****", command)
            x = os.system(command)
            if x:
                print(x)
                exit(x)


if __name__ == "__main__":
	main()
