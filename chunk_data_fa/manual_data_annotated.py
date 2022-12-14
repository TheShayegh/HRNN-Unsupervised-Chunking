import pickle as pkl
import pandas as pd

original_sentences = pkl.load(open('persian_sentences.pickle', 'rb'))
original_tags = pkl.load(open('persian_labels.pickle', 'rb'))
annotated_df = pd.read_csv('ChunkingDataPrepration - Data.csv')

unannotated = annotated_df['Is not valid'].isna() * annotated_df['Annotations'].isna()
annotated = ~unannotated
annotated_sentences = annotated_df.loc[annotated, 'sentence'].values

training_tokens = []
training_tags = []
for sentence, tag in zip(original_sentences, original_tags):
    if ' '.join(sentence) in annotated_sentences:
        continue
    if len(set(tag))==1:
        continue
    training_tokens.append(sentence)
    new_tag = ['3' if t=='O' else '1' if t=='I-NP' else '2' for t in tag]
    training_tags.append(new_tag)

training_data = list(zip(training_tokens, training_tags))
pkl.dump(training_data, open("./train.pkl", "wb"))

validation_tokens = []
validation_tags = []
validation_set = annotated_df[~annotated_df['Annotations'].isna()]
for sentence, tag in validation_set[['sentence', 'Annotations']].values:
    if len(set(tag))==1:
        continue
    validation_tokens.append(sentence.split())
    new_tag = ['3' if t=='O' else '1' if t=='I' else '2' for t in tag.split()]
    validation_tags.append(new_tag)

validation_data = list(zip(validation_tokens, validation_tags))
pkl.dump(validation_data, open("validation.pkl", 'wb'))
pkl.dump([tag.split() for tag in validation_set['Annotations']], open("validation_tag.pkl", 'wb'))