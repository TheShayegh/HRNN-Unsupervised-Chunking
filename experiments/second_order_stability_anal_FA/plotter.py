import yaml
import sys
import os
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(1, '../../')
from library.HRNN import eval_conll2000
from eval_heuristic import eval_hu

N = 10
M = 10

def main():
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    home = config['home']

    test_metrics = {'fscore': {i: [] for i in range(N)}, 'acc': {i: [] for i in range(N)}}
    init_metrics = {'fscore': {}, 'acc': {}}
    single_words = {}

    for i in range(N):
        for j in range(M):

            fscore, acc = eval_conll2000(
                open(config['test_output_path'].format(i, j), 'r').read(),
                eval_conll_path="../../library/eval_conll.pl"
            )
            test_metrics['fscore'][i].append(fscore)
            test_metrics['acc'][i].append(acc)
        
    for i in range(N):
        
        test_data = pickle.load(open('../../'+config['target_path'].format(config['home'], 'test', i), 'rb'))
        test_tags_gt = pickle.load(open('../../'+config['test_true_tags'], 'rb'))
        fscore, acc = eval_hu(test_data, test_tags_gt, eval_conll_path="../../library/eval_conll.pl")
        init_metrics['fscore'][i] = fscore
        init_metrics['acc'][i] = acc

    sw_fscore, sw_acc = eval_hu(test_data, test_tags_gt, single_words=True, eval_conll_path="../../library/eval_conll.pl")
    single_words['fscore'] = sw_fscore
    single_words['acc'] = sw_acc

    

    def plot(metric):
        x = sns.boxplot(pd.DataFrame(test_metrics[metric]))
        for i in range(N):
            x.text(i, min(test_metrics[metric][i])-.5, init_metrics[metric][i], 
                horizontalalignment='center',size='x-small',color='k',weight='semibold')
        x.set_ylim(min([min(test_metrics[metric][i]) for i in range(N)])-1, None)
        x.axhline(single_words[metric], linestyle='--', color='k')
        x.set_title(f'signle word heuristic {metric}: {single_words[metric]}')
        x.get_figure().savefig(f"test_{metric}.png")
        plt.clf()

    plot('fscore')
    plot('acc')

if __name__ == "__main__":
	main()
