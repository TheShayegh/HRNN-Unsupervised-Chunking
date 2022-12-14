import torch
from tqdm import tqdm
import pickle
from library.utils import build_vocab, data_padding
from torchtext.data import BucketIterator
from library.HRNN import eval_conll2000

B_PROB = 1
# validation_tag = "conll_data_en/validation_tag.pkl"
# validation = "conll_data_en/validation.pkl"
validation_tag = "chunk_data_fa/validation_tag.pkl"
validation = "chunk_data_fa/validation.pkl"

BATCH_SIZE=1
def make_bucket_iterator(
    data,
    device: torch.device = 'cpu',
):
    bucket_iterator = BucketIterator(
        data, 
        batch_size=BATCH_SIZE,
        sort_key=lambda x: np.count_nonzero(x[0]),
        sort=False, 
        shuffle=False,
        sort_within_batch=False,
        device=device,
    )
    bucket_iterator.create_batches()
    return bucket_iterator

def validate(
    p,
    data,
    true_tags,
) -> str:
    bucket_iterator = make_bucket_iterator(data)
    output = ""
    for batch, true_tag in tqdm(zip(bucket_iterator.batches, true_tags), total=len(bucket_iterator)):
        output += "x y B B\n"
        for i, true_label in enumerate(true_tag[1:]):
            predicted_label = "B" if torch.rand(1)[0]>p else "I"
            output += f"x y {true_label} {predicted_label}\n"
    return output


validation_true_tags = pickle.load(open(validation_tag, "rb"))
validation_data = pickle.load(open(validation, "rb"))
word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = build_vocab(validation_data)
validation_tokens, validation_tags, validation_msl = data_padding(validation_data, word_to_ix, tag_to_ix, device='cpu')
data = list(zip(validation_tokens, validation_tags))

validation_output = validate(1-B_PROB, data, validation_true_tags)
fscore, acc = eval_conll2000(validation_output)
print( " __________________________________")
print(f"|     F1:       {fscore}")
print(f"|     Accuracy: {acc}")
print( "|__________________________________")