import pickle

# PATH = "HRNNdata_due/train.pkl"
# TRUE_TAGS_PATH = "conll2003_due/train_tags.pkl"
# TARGET_PATH = "HRNNdata_due/trusted_finetune.pkl"
PATH = "HRNNdata_en_original/train.pkl"
TRUE_TAGS_PATH = "conll2000_en/train_tags.pkl"
TARGET_PATH = "HRNNdata_en_original/trusted_finetune.pkl"
SIZE = 100


data = pickle.load(open(PATH, 'rb'))
tokens = [sentence for sentence,_ in data]
tags_gt = pickle.load(open(TRUE_TAGS_PATH, 'rb'))

line_no = 0
tags = []
for sentence, tag_set in zip(tokens[:SIZE], tags_gt[:SIZE]):
    sentence_tags = []
    line_no += 1
    for token, tag in zip(sentence[:-1], tag_set[1:]):
        sentence_tags.append('2' if tag=='B' else '1')
        line_no += 1
    sentence_tags.append('1')
    tags.append(sentence_tags)

targets = [
    [sentence_tokens, sentence_tags]
    for sentence_tags, sentence_tokens in zip(tags, tokens)
]

pickle.dump(targets, open(TARGET_PATH, 'wb'))