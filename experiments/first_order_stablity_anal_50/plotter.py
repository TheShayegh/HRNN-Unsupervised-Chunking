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

    # train_loss = []
    # validation_metrics = []
    test_metrics = {'fscore': {l: [] for l in config["languages"]}, 'acc': {l: [] for l in config["languages"]}}

    for i in range(10):
        # train_loss_path = config['train_loss'].format(i)
        # validation_metrics_path = config['validation_metrics'].format(i)

        # train_loss.append(pickle.load(open(train_loss_path, 'rb')))
        # validation_metrics.append(pickle.load(open(validation_metrics_path, 'rb')))

        for lang in config["languages"]:
            fscore, acc = eval_conll2000(
                open(config['test_output_path'].format(i, lang), 'r').read(),
                eval_conll_path="../../library/eval_conll.pl"
            )
            test_metrics['fscore'][lang].append(fscore)
            test_metrics['acc'][lang].append(acc)

    # for tl in train_loss:
    #     x = sns.lineplot(tl, color='b', alpha=.3)
    # x.get_figure().savefig("train_loss.png")
    # plt.clf()
    # for vm in validation_metrics:
    #     x = sns.lineplot(vm['loss'], color='b', alpha=.3)
    # x.get_figure().savefig("validation_loss.png")
    # plt.clf()
    # for vm in validation_metrics:
    #     x = sns.lineplot(vm['fscore'], color='b', alpha=.3)
    # x.get_figure().savefig("validation_fscore.png")
    # plt.clf()
    # for vm in validation_metrics:
    #     x = sns.lineplot(vm['acc'], color='b', alpha=.3)
    # x.get_figure().savefig("validation_acc.png")
    # plt.clf()
    sns.boxplot(pd.DataFrame(test_metrics['fscore'])).get_figure().savefig("test_fscore.png")
    plt.clf()
    sns.boxplot(pd.DataFrame(test_metrics['acc'])).get_figure().savefig("test_acc.png")
    plt.clf()

if __name__ == "__main__":
	main()
