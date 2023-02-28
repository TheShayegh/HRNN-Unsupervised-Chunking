import pickle
from tqdm import tqdm
from library.HRNN import validation_output
from library.HRNN import eval_conll2000

def eval_hu(test_data, test_tags_gt, single_words=False, **args):
    output = ""
    for i, (pair, true_tag) in tqdm(enumerate(zip(test_data, test_tags_gt)), total=len(test_data)):
        if single_words:
            ind = [1 for t in pair[1]]
        else:
            ind = [t=='2' for t in pair[1]]
        vo = validation_output(ind, true_tag)
        output += vo
    fscore, acc = eval_conll2000(output, **args)
    return fscore, acc


if __name__ == "__main__":
    # TEST_PATH = "experiments/second_order_hiddendim_anal_FA/test_predicted_0.pkl"
    TEST_PATH = "HRNNdata_fa/test.pkl"
    TEST_TRUE_TAGS_PATH = "HRNNdata_fa/test_tag.pkl"

    test_data = pickle.load(open(TEST_PATH, 'rb'))
    test_tags_gt = pickle.load(open(TEST_TRUE_TAGS_PATH, 'rb'))#[:len(test_data)]
    print(len(test_data), len(test_tags_gt))
    fscore, acc = eval_hu(test_data, test_tags_gt)
    print( " __________________________________")
    print(f"| Test:")
    print(f"|     F1:       {fscore}")
    print(f"|     Accuracy: {acc}")
    print( "|__________________________________")