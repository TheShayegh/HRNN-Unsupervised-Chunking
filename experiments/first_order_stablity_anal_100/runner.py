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
        print(f"*** Run {i}th")
        config_dimed = deepcopy(config)
        del config_dimed['languages']
        config_dimed['best_model_path'] = config['best_model_path'].format(i)

        with open("_config.yml", 'w') as f:
            f.write(yaml.dump(config_dimed))

        x = os.system(f'cd ../../ && CUDA_VISIBLE_DEVICES={sys.argv[2]} python3 train_script.py {home}_config.yml')
        if x:
            print(x)
            exit(x)

        for lang, test_data, test_true_tags in zip(config["languages"], config["test_data"], config["test_true_tags"]):
            # if lang!='FA':
            #     continue
        
            print(f"*** language = {lang}")
            config_dimed_langed = deepcopy(config_dimed)
            config_dimed_langed['test_data'] = test_data
            config_dimed_langed['test_true_tags'] = test_true_tags
            config_dimed_langed['test_output_path'] = config['test_output_path'].format(i, lang)
            config_dimed_langed['test_embeddings'] = config['test_embeddings'].format(lang)

            with open("_config.yml", 'w') as f:
                f.write(yaml.dump(config_dimed_langed))

            x = os.system(f'cd ../../ && CUDA_VISIBLE_DEVICES={sys.argv[2]} python3 test_script.py {home}_config.yml')
            if x:
                print(x)
                exit(x)


if __name__ == "__main__":
	main()
