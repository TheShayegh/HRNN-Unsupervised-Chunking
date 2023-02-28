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

N = 3

def main():
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    home = config['home']

    test_metrics = {
        'fscore': [{l: [] for l in config["languages"]} for i in range(N)],
        'acc': [{l: [] for l in config["languages"]} for i in range(N)],
    }

    for hidden_dim in tqdm(config['hidden_dim']):
        for i in range(N):

            for lang in config["languages"]:
                fscore, acc = eval_conll2000(
                    open(config['test_output_path'].format(hidden_dim, lang, i), 'r').read(),
                    eval_conll_path="../../library/eval_conll.pl"
                )
                test_metrics['fscore'][i][lang].append(fscore)
                test_metrics['acc'][i][lang].append(acc)
    
    df = lambda metric, i: pd.DataFrame(test_metrics[metric][i], index=config['hidden_dim'])

    for i in range(N):
        x = sns.lineplot(df('fscore', i), alpha=(i+2)/8, legend=False)
    x = sns.lineplot(sum([df('fscore', i) for i in range(N)])/N)
    x.get_figure().savefig("test_fscore.png")
    plt.clf()

    for i in range(N):
        x = sns.lineplot(df('acc', i), alpha=(i+2)/8, legend=False)
    x = sns.lineplot(sum([df('acc', i) for i in range(N)])/N)
    x.get_figure().savefig("test_acc.png")
    plt.clf()

if __name__ == "__main__":
	main()
