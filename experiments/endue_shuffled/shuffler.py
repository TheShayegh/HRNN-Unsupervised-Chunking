from random import shuffle
import pickle

DUE_TRAIN_PATH = '../due_trained_on_en_output_mullingbert/train.pkl'
DUE_VAL_PATH = '../due_trained_on_en_output_mullingbert/val.pkl'
DUE_VALTAG_PATH = '../../HRNNdata_due/val_tag.pkl'
EN_TRAIN_PATH = '../../HRNNdata_en/train.pkl'
EN_VAL_PATH = '../../HRNNdata_en/val.pkl'
EN_VALTAG_PATH = '../../HRNNdata_en/val_tag.pkl'

TARGET_TRAIN_PATH = './train.pkl'
TARGET_VAL_PATH = './val.pkl'
TARGET_VALTAG_PATH = './val_tag.pkl'

due_train = pickle.load(open(DUE_TRAIN_PATH, 'rb'))
en_train = pickle.load(open(EN_TRAIN_PATH, 'rb'))
train = due_train+en_train
shuffle(train)
pickle.dump(train, open(TARGET_TRAIN_PATH, 'wb'))

due_val = pickle.load(open(DUE_VAL_PATH, 'rb'))
en_val = pickle.load(open(EN_VAL_PATH, 'rb'))
due_val_tag = pickle.load(open(DUE_VALTAG_PATH, 'rb'))
en_val_tag = pickle.load(open(EN_VALTAG_PATH, 'rb'))
due_val = list(zip(due_val, due_val_tag))
en_val = list(zip(en_val, en_val_tag))
val = due_val+en_val
shuffle(val)
val, val_tag = zip(*val)
val, val_tag = list(val), list(val_tag)
pickle.dump(val, open(TARGET_VAL_PATH, 'wb'))
pickle.dump(val_tag, open(TARGET_VALTAG_PATH, 'wb'))