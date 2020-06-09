from collections import defaultdict
#from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data.dataloader import DataLoader


UNKNOWN_TOKEN = "<unk>"
ROOT_TOKEN = "<root>"  # use this if you are padding your batches and want a special token for ROOT
SPECIAL_TOKENS = [ROOT_TOKEN, UNKNOWN_TOKEN]


def split(string, delimiters):
    """
        Split strings according to delimiters
        :param string: full sentence
        :param delimiters string: characters for spliting
            function splits sentence to words
    """
    delimiters = tuple(delimiters)
    stack = [string, ]

    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i + j, _substring)

    return stack

def get_vocabs(list_of_paths):
    """
        Extract vocabs from given datasets. Return a word2idx and tag2idx.
        :param file_paths: a list with a full path for all corpuses
            Return:
              - word2idx
              - tag2idx
    """
    word_dict = defaultdict(int)
    pos_dict = defaultdict(int)
    # lengths of mappings
    len_word2idx = 0
    len_pos2idx = 0
    # Add special token to word2idx
    word_dict[ROOT_TOKEN] = len_word2idx
    len_word2idx += 1
    word_dict[UNKNOWN_TOKEN] = len_word2idx
    len_word2idx += 1
    # Add special token to pos2idx
    pos_dict[ROOT_TOKEN] = len_pos2idx
    len_pos2idx += 1
    pos_dict[UNKNOWN_TOKEN] = len_pos2idx
    len_pos2idx += 1
    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                splited_words = split(line, (' ', '\t', '\n'))  # len 10
                del splited_words[-1]
                if(len(splited_words)!=10):
                    continue
                word, pos = splited_words[1], splited_words[3]
                if word not in word_dict:
                    word_dict[word] = len_word2idx
                    len_word2idx += 1
                if pos not in pos_dict:
                    pos_dict[pos] = len_pos2idx
                    len_pos2idx += 1
    return word_dict, pos_dict


class PosDataReader:
    def __init__(self, file, word_dict, pos_dict):
        self.file = file
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.sentences = []
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        with open(self.file, 'r') as f:
            cur_sentence = []
            for line in f:
                splited_words = split(line, (' ', '\t', '\n'))
                del splited_words[-1]
                # splited_words is of length 10 if we are still in the same sentence
                if len(splited_words) == 10:
                    cur_word, cur_tag, token_head = splited_words[1], splited_words[3], splited_words[6]
                    cur_sentence.append((cur_word, cur_tag, token_head))
                else:
                    self.sentences.append(cur_sentence)
                    cur_sentence = []

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)


# class that creates the data set
class PosDataset(Dataset):
    def __init__(self, word_dict, pos_dict, dir_path: str, subset: str, word_embeddings=None):
        super().__init__()
        self.subset = subset  # One of the following: [train, test]
        self.file = dir_path + subset + ".labeled"
        self.datareader = PosDataReader(self.file, word_dict, pos_dict)
        self.vocab_size = len(self.datareader.word_dict)  # how many words in corpus
        self.word2idx = word_dict
        # if word_embeddings:  # if we have a word embeding -> embed
        #     self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = word_embeddings
        # else:  # use pre-trained (glove)
        #     self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings(self.datareader.word_dict)

        # self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab(self.datareader.pos_dict)
        self.pos_idx_mappings = pos_dict

        self.root_idx = self.word2idx.get(ROOT_TOKEN)
        self.unknown_idx = self.word2idx.get(UNKNOWN_TOKEN)
        self.sentences_dataset = self.convert_sentences_to_dataset()

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, head_idx = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, head_idx

    # @staticmethod
    # def init_word_embeddings(word_dict):
    #     glove = Vocab(Counter(word_dict), vectors="glove.6B.300d", specials=SPECIAL_TOKENS)
    #     return glove.stoi, glove.itos, glove.vectors
    #
    # def get_word_embeddings(self):
    #     return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    # def init_pos_vocab(self, pos_dict):
    #     idx_pos_mappings = sorted([self.word2idx.get(token) for token in SPECIAL_TOKENS])
    #     pos_idx_mappings = {self.word2idx[idx]: idx for idx in idx_pos_mappings}
    #     for i, pos in enumerate(sorted(pos_dict.keys())):
    #         pos_idx_mappings[str(pos)] = int(i + len(SPECIAL_TOKENS))
    #         idx_pos_mappings.append(str(pos))
    #     print(pos_idx_mappings.get(ROOT_TOKEN))
    #     return pos_idx_mappings, idx_pos_mappings

    # def get_pos_vocab(self):
    #     return self.pos_idx_mappings, self.idx_pos_mappings

    def convert_sentences_to_dataset(self):
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_head_idx_list = list()
        for sentence_idx, sentence in enumerate(self.datareader.sentences):
            words_idx_list = []
            pos_idx_list = []
            head_idx_list = []
            # add root
            words_idx_list.append(self.root_idx)
            pos_idx_list.append(self.pos_idx_mappings.get(ROOT_TOKEN))
            head_idx_list.append(-1)    # align with chu-liu edmonds output
            for word, pos, head_token in sentence:
                if word not in self.word2idx:
                    words_idx_list.append(self.unknown_idx)
                else:
                    words_idx_list.append(self.word2idx.get(word))
                pos_idx_list.append(self.pos_idx_mappings.get(pos))
                head_idx_list.append(int(head_token))
            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_head_idx_list.append(torch.tensor(head_idx_list, dtype=torch.long, requires_grad=False))
        return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_word_idx_list, sentence_pos_idx_list, sentence_head_idx_list))}


data_dir = "HW2-files/"
path_train = data_dir + "train.labeled"
print("path_train -", path_train)
path_test = data_dir + "test.labeled"
print("path_test -", path_test)

paths_list = [path_train, path_test]
word_dict, pos_dict = get_vocabs(paths_list)
train = PosDataset(word_dict, pos_dict, data_dir, 'train')
train_dataloader = DataLoader(train, shuffle=True)
test = PosDataset(word_dict, pos_dict, data_dir, 'test')
test_dataloader = DataLoader(test, shuffle=False)


a = next(iter(train_dataloader))
#a[0] -> word - idx of a sentence
#a[1] -> pos - idx of a sentence
#a[2] -> head token per sentence
assert len(a[0])==len(a[1])==len(a[2])

for batch_idx, input_data in enumerate(train_dataloader):
    if batch_idx>0:
        break
    sentence, pos, heads = input_data
    print(sentence.shape)
    print(pos.shape)
    print(heads.shape)