import pickle
from tqdm import tqdm
from library.HRNN import validation_output
from library.HRNN import eval_conll2000

def eval_hu(test_data, test_tags_gt, **args):
    output = ""
    for i, (pair, true_tag) in tqdm(enumerate(zip(test_data, test_tags_gt)), total=len(test_data)):
        ind = [t=='2' for t in pair[1]]
        vo = validation_output(ind, true_tag)
        output += vo
    fscore, acc = eval_conll2000(output, **args)
    return fscore, acc


if __name__ == "__main__":

    # TEST_PATH = "HRNNdata_due/test.pkl.predicted.pkl"
    # TEST_PATH = "HRNNdata_due/test.pkl"
    # TEST_TRUE_TAGS_PATH = "HRNNdata_due/test_tag.pkl"

    TEST_PATH = "HRNNdata_en_original/test.pkl"
    TEST_TRUE_TAGS_PATH = "HRNNdata_en_original/test_tag.pkl"

    # TEST_PATH = "HRNNdata_due/trusted_finetune.pkl"
    # TEST_TRUE_TAGS_PATH = "conll2003_due/train_tags.pkl"

    # TEST_PATH = "HRNNdata_en_original/trusted_finetune.pkl"
    # TEST_TRUE_TAGS_PATH = "conll2000_en/train_tags.pkl"

    # TEST_PATH = "HRNNdata_fa/test.pkl.predicted.pkl"
    # TEST_PATH = "experiments/second_order_stability_anal_FA/test_predicted0.pkl"
    # TEST_TRUE_TAGS_PATH = "HRNNdata_fa/test_tag.pkl"

    test_data = pickle.load(open(TEST_PATH, 'rb'))
    test_tags_gt = pickle.load(open(TEST_TRUE_TAGS_PATH, 'rb'))#[:len(test_data)]
    fscore, acc = eval_hu(test_data, test_tags_gt)
    print( " __________________________________")
    print(f"| Test:")
    print(f"|     F1:       {fscore}")
    print(f"|     Accuracy: {acc}")
    print( "|__________________________________")