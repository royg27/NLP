import basic_model
import advanced_model
import utils
from torch.utils.data.dataset import Dataset, TensorDataset
from chu_liu_edmonds import *
from os import path
import torch
from torchtext.vocab import Vocab
from collections import Counter

from utils import *


def tagger(model, data_loader):
    tags = []
    for batch_idx, input_data in enumerate(data_loader):
        with torch.no_grad():
            words_idx_tensor, pos_idx_tensor, heads_tensor = input_data
            tag_scores = model(words_idx_tensor, pos_idx_tensor)
            predicted_mst, _ = decode_mst(energy=tag_scores.detach().cpu(), length=tag_scores.shape[0], has_labels=False)
            tags.append(predicted_mst[1:])
    return tags

# create data sets
data_dir = "HW2-files/"
path_train = data_dir + "train.labeled"
path_test = data_dir + "test.labeled"
paths_list = [path_train, path_test]
word_cnt, word_dict, pos_dict = utils.get_vocabs(paths_list)
train = utils.PosDataset(word_cnt, word_dict, pos_dict, data_dir, 'train')
train_dataloader = utils.DataLoader(train, shuffle=True)
test = utils.PosDataset(word_cnt, word_dict, pos_dict, data_dir, 'test')
test_dataloader = utils.DataLoader(test, shuffle=False)
word_vocab_size = len(train.word2idx)
tag_vocab_size = len(train.pos_idx_mappings)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# create and load trained model
base_model = basic_model.DnnDependencyParser(basic_model.WORD_EMBEDDING_DIM, basic_model.POS_EMBEDDING_DIM, 
                                              basic_model.HIDDEN_DIM, word_vocab_size, tag_vocab_size).to(device)
if path.exists("./basic_model_best_params"):
    print("loading model")
    use_cuda = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if use_cuda else 'cpu')  # 'cpu' in this case
    base_model.load_state_dict(torch.load("./basic_model_best_params", map_location=DEVICE))
    # basic_model.load_state_dict(torch.load(PATH))
    print("acc on test = ", basic_model.evaluate(base_model, test_dataloader))
else:
    print("model not found")
# inference
path_comp = data_dir + "comp.unlabeled"

comp = utils.PosDataset(word_cnt, word_dict, pos_dict, data_dir, 'comp')
comp_dataloader = utils.DataLoader(comp, shuffle=False)   # comp data loader
head_tokens_tags = tagger(base_model, comp_dataloader)
utils.comp_tagger(path_comp, "comp_m1_204506349.labeled", head_tokens_tags)

# create and load trained model
glove = Vocab(Counter(word_dict), vectors="glove.6B.300d", specials=SPECIAL_TOKENS)
adv_model = advanced_model.AdvancedDnnDependencyParser(advanced_model.WORD_EMBEDDING_DIM, advanced_model.POS_EMBEDDING_DIM,
                                              advanced_model.HIDDEN_DIM, word_vocab_size, tag_vocab_size,
                                              num_lst_layers = advanced_model.LSTM_LAYERS, word_embedding_table=glove.vectors).to(device)
if path.exists("./advanced_model_best_params"):
    print("loading model")
    use_cuda = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if use_cuda else 'cpu')  # 'cpu' in this case
    adv_model.load_state_dict(torch.load("./advanced_model_best_params", map_location=DEVICE))
    print("acc on test = ", advanced_model.evaluate(adv_model, test_dataloader))
else:
    print("model not found")
# inference
path_comp = data_dir + "comp.unlabeled"

comp = utils.PosDataset(word_cnt, word_dict, pos_dict, data_dir, 'comp')
comp_dataloader = utils.DataLoader(comp, shuffle=False)   # comp data loader
head_tokens_tags = tagger(adv_model, comp_dataloader)
utils.comp_tagger(path_comp, "comp_m2_204506349.labeled", head_tokens_tags)
