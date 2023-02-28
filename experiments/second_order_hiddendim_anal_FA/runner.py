import yaml
import sys
import os
from copy import deepcopy

N = 3
M = 3

def main():
    print(f"*** CUDA_VISIBLE_DEVICES = {sys.argv[2]}\n")
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    home = config['home']

    for i in range(N):
        for data in ['train', 'validation', 'test']:
            config_idata = deepcopy(config)
            config_idata['best_model_path'] = config['source_model_path'].format(i)
            config_idata['use_data'] = config[data+'_data']
            config_idata['use_embeddings'] = 'source_'+config[data+'_embeddings']
            config_idata['target_path'] = config['target_path'].format(home, data, i)

            with open(config['source_config']) as f:
                source_config = yaml.load(f, Loader=yaml.FullLoader)
                
                for item in ['device', 'embedding_dim', 'hidden_dim', 'learning_rate', 'warmup', 'epocs',
                    'embedding_mode', 'train_embedding_path', 'val_embedding_path', 'test_embedding_path',
                    'embedding_token_heuristic', 'bert_pretrained_weights', 'validation_mode', 'test_mode',
                    'enforced_mode', 'use_embedding_path', 'use_mode']:
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


        for j in range(M):

            for hidden_dim in config['hidden_dim']:
            
                # if os.path.exists(config['test_output_path'].format(hidden_dim, i, j)):
                #     continue
                
                print(f"*** i = {i}, j = {j}, hidden_dim = {hidden_dim}")
                config_ij = deepcopy(config)
                config_ij['hidden_dim'] = hidden_dim
                config_ij['test_output_path'] = config['test_output_path'].format(hidden_dim, i, j)
                config_ij['best_model_path'] = config['best_model_path'].format(hidden_dim, i, j)
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
