import os
import numpy as np
import re
from scipy.sparse import csr_matrix
import settings

class textProcessor:
    def __init__(self, path_file_name):
        self.file = open(path_file_name,'r')
        self.lines = self.file.readlines()
        self.sentences = []
        self.tags = []
        self.histories = []

        self.feature_100 = {}
        self.feature_101 = {}
        self.feature_102 = {}
        self.feature_103 = {}
        self.feature_104 = {}
        self.feature_105 = {}

        self.words_set = set()
        self.tags_set = set('*')
        # length of feature vector
        self.f_length = -1

    def preprocess(self):
        # Split all the sentences and all the words and tags
        for line in self.lines:
            temp = re.split(' |\n', line)
            temp = temp[0:-1]
            split_sentence = []
            split_tags = []
            for str in temp:
                t = re.split('_', str)
                split_sentence.append(t[0])
                split_tags.append(t[1])
            assert len(split_sentence) == len(split_tags)
            self.sentences.append(split_sentence)
            self.tags.append(split_tags)

        # Create all the histories needed
        word_count = 0
        for sentence_idx, sentence in enumerate(self.sentences):
            sentence_history = []
            for word_idx, word in enumerate(sentence):
                word_count += 1
                # history = ( t-2,t-1,t,w )
                t = self.tags[sentence_idx][word_idx]
                t_1 = self.tags[sentence_idx][word_idx-1] if word_idx > 0 else '*'
                t_2 = self.tags[sentence_idx][word_idx-2] if word_idx > 1 else '*'
                curr_history = (t_2, t_1, t, word)
                sentence_history.append(curr_history)
                self.words_set.add(word)
                self.tags_set.add(t)
            self.histories.append(sentence_history)

        self.fill_feature_100_dictionary()
        self.fill_feature_101_102_105_dictionary()
        self.fill_feature_103_dictionary()
        self.fill_feature_104_dictionary()

        self.f_length = len(self.feature_100) + len(self.feature_101) + len(self.feature_102) + len(self.feature_103) +\
                        len(self.feature_104) + len(self.feature_105)


    def fill_feature_100_dictionary(self):
        idx = 0
        for word in self.words_set:
            for tag in self.tags_set:
                self.feature_100[(word, tag)] = idx
                idx += 1

    def fill_feature_101_102_105_dictionary(self):
        for idx,tag in enumerate(self.tags_set):
            self.feature_101[tag] = idx
            self.feature_102[tag] = idx
            self.feature_105[tag] = idx

    def fill_feature_103_dictionary(self):
        idx = 0
        for first_tag in self.tags_set:
            for sec_tag in self.tags_set:
                for thrd_tag in self.tags_set:
                    key = (first_tag, sec_tag, thrd_tag)
                    self.feature_103[key] = idx
                    idx += 1

    def fill_feature_104_dictionary(self):
        idx = 0
        for first_tag in self.tags_set:
            for sec_tag in self.tags_set:
                key = (first_tag, sec_tag)
                self.feature_104[key] = idx
                idx += 1

    def generate_feature_vector(self, history):
        # history = (t-2,t-1,t,w)
        t_2 = history[0]
        t_1 = history[1]
        tag = history[2]
        word = history[3]

        # f100
        f100_idx = self.feature_100[(word, tag)]
        f100 = np.zeros(len(self.feature_100))
        f100[f100_idx] = 1

        # f101
        f101_idx = self.feature_101[tag]
        f101 = np.zeros(len(self.feature_101))
        if word[-3:] == "ing":
            f101[f101_idx] = 1

        # f102
        f102_idx = self.feature_102[tag]
        f102 = np.zeros(len(self.feature_102))
        if word[:3] == "pre":
            f102[f102_idx] = 1

        # f103
        f103_idx = self.feature_103[(t_2,t_1,tag)]
        f103 = np.zeros(len(self.feature_103))
        f103[f103_idx] = 1

        # f104
        f104_idx = self.feature_104[(t_1,tag)]
        f104 = np.zeros(len(self.feature_104))
        f104[f104_idx] = 1

        # f105
        f105_idx = self.feature_105[tag]
        f105 = np.zeros(len(self.feature_105))
        f105[f105_idx] = 1

        final_feature_vector = np.concatenate((f100, f101, f102, f103, f104, f105))
        return final_feature_vector


    def generate_expected_count_features(self, history):
        # history = (t-2,t-1,t,w)
        t_2 = history[0]
        t_1 = history[1]
        tag = history[2]
        word = history[3]

        expected_count_features = []
        for possible_tag in self.tags_set:
            possible_history = (t_2, t_1, possible_tag, word)
            expected_count_features.append(self.generate_feature_vector(possible_history))
        assert len(expected_count_features) == len(self.tags_set)
        if settings.use_vectorized_sparse:
            return csr_matrix(expected_count_features)
        else:
            return expected_count_features
