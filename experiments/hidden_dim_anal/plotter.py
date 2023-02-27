import yaml
import sys
import os
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.insert(1, '../../')
from library.HRNN import eval_conll2000
from eval_heuristic import eval_hu

N = 3
hu_langs = ['TrainEN']

def main():
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    home = config['home']

    # train_loss = {}
    # validation_metrics = {}
    test_metrics = {
        'fscore': [{l: [] for l in config["languages"]} for i in range(N)],
        'acc': [{l: [] for l in config["languages"]} for i in range(N)],
    }
    for lang in hu_langs:
        for i in range(N):
            test_metrics['fscore'][i][lang+' compared to the heuristic'] = []
            test_metrics['acc'][i][lang+' compared to the heuristic'] = []

    for hidden_dim in tqdm(config['hidden_dim']):
        for i in range(N):
            # train_loss_path = config['train_loss'].format(hidden_dim)
            # validation_metrics_path = config['validation_metrics'].format(hidden_dim)

            # train_loss[hidden_dim] = pickle.load(open(train_loss_path, 'rb'))
            # validation_metrics[hidden_dim] = pickle.load(open(validation_metrics_path, 'rb'))

            for lang in config["languages"]:
                fscore, acc = eval_conll2000(
                    open(config['test_output_path'].format(hidden_dim, lang, i), 'r').read(),
                    eval_conll_path="../../library/eval_conll.pl"
                )
                test_metrics['fscore'][i][lang].append(fscore)
                test_metrics['acc'][i][lang].append(acc)
            
            for lang in hu_langs:
                hu_data = pickle.load(open('../../'+config['test_data'][config['languages'].index(lang)], 'rb'))
                output = open(config['test_output_path'].format(hidden_dim, lang, i), 'r').read().strip().split('\n')
                gt = []
                for _,hu in hu_data:
                    gt_record = output[:len(hu)]
                    output = output[len(hu):]
                    gt_record = [gtline.strip()[-1] for gtline in gt_record]
                    gt.append(gt_record)
                fscore, acc = eval_hu(hu_data, gt, eval_conll_path="../../library/eval_conll.pl")
                test_metrics['fscore'][i][lang+' compared to the heuristic'].append(fscore)
                test_metrics['acc'][i][lang+' compared to the heuristic'].append(acc)


    # sns.lineplot(pd.DataFrame(train_loss)).get_figure().savefig("train_loss.png")
    # plt.clf()
    # sns.lineplot(pd.DataFrame({k: v['loss'] for k,v in validation_metrics.items()})).get_figure().savefig("validation_loss.png")
    # plt.clf()
    # sns.lineplot(pd.DataFrame({k: v['fscore'] for k,v in validation_metrics.items()})).get_figure().savefig("validation_fscore.png")
    # plt.clf()
    # sns.lineplot(pd.DataFrame({k: v['acc'] for k,v in validation_metrics.items()})).get_figure().savefig("validation_acc.png")
    # plt.clf()
    for i in range(N):
        x = sns.lineplot(pd.DataFrame(test_metrics['fscore'][i], index=config['hidden_dim']), alpha=.3, legend=False)
    x = sns.lineplot(sum([pd.DataFrame(test_metrics['fscore'][i], index=config['hidden_dim']) for i in range(N)])/N)
    x.get_figure().savefig("test_fscore.png")
    plt.clf()
    for i in range(N):
        x = sns.lineplot(pd.DataFrame(test_metrics['acc'][i], index=config['hidden_dim']), alpha=.3, legend=False)
    x = sns.lineplot(sum([pd.DataFrame(test_metrics['acc'][i], index=config['hidden_dim']) for i in range(N)])/N)
    x.get_figure().savefig("test_acc.png")
    plt.clf()

if __name__ == "__main__":
	main()
