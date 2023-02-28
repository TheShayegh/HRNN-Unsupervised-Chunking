import yaml
import sys
import os
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(1, '../../')
from library.HRNN import eval_conll2000

def main():
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    home = config['home']

    test_metrics = {'fscore': {l: [] for l in config["languages"]}, 'acc': {l: [] for l in config["languages"]}}

    for i in range(10):
        for lang in config["languages"]:
            fscore, acc = eval_conll2000(
                open(config['test_output_path'].format(i, lang), 'r').read(),
                eval_conll_path="../../library/eval_conll.pl"
            )
            test_metrics['fscore'][lang].append(fscore)
            test_metrics['acc'][lang].append(acc)

    sns.boxplot(pd.DataFrame(test_metrics['fscore'])).get_figure().savefig("test_fscore.png")
    plt.clf()
    sns.boxplot(pd.DataFrame(test_metrics['acc'])).get_figure().savefig("test_acc.png")
    plt.clf()

if __name__ == "__main__":
	main()
