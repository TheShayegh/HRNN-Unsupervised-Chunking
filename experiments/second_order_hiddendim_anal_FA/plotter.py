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

M = 3
N = 3

def main():
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    home = config['home']

    default_dict = lambda N, M: {i: {j: [] for j in range(M)} for i in range(N)}
    test_metrics = {'fscore': default_dict(N, M), 'acc': default_dict(N, M)}
    init_metrics = {'fscore': {}, 'acc': {}}

    for i in range(N):
        for j in range(M):
            
            for hidden_dim in config['hidden_dim']:
                fscore, acc = eval_conll2000(
                    open(config['test_output_path'].format(hidden_dim, i, j), 'r').read(),
                    eval_conll_path="../../library/eval_conll.pl"
                )
                test_metrics['fscore'][j][i].append(fscore)
                test_metrics['acc'][j][i].append(acc)


    for i in range(N):
        
        test_data = pickle.load(open('../../'+config['target_path'].format(config['home'], 'test', i), 'rb'))
        test_tags_gt = pickle.load(open('../../'+config['test_true_tags'], 'rb'))
        fscore, acc = eval_hu(test_data, test_tags_gt, eval_conll_path="../../library/eval_conll.pl")
        init_metrics['fscore'][i] = fscore
        init_metrics['acc'][i] = acc

    df = lambda metric, j: pd.DataFrame(test_metrics[metric][j], index=config['hidden_dim'])

    def plot(metric):
        for j in range(M):
            x = sns.lineplot(df(metric, j), alpha=.2, legend=False)
        x = sns.lineplot(sum([df(metric, j) for j in range(M)])/M)
        for i in range(N):
            x.axhline(init_metrics[metric][i], linestyle=['-','--',':'][i], color=['b','orange','g'][i])
        x.set_ylim(min(init_metrics[metric].values())-1, None)
        x.get_figure().savefig(f"test_{metric}.png")
        plt.clf()

    plot('fscore')
    plot('acc')

if __name__ == "__main__":
	main()
