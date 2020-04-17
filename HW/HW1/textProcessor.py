import os
import numpy as np
import re
from scipy.sparse import csr_matrix
import settings
from collections import OrderedDict

class textProcessor:
    def __init__(self, path_file_name, thr=3):
        self.thr = thr
        self.file = open(path_file_name,'r')
        self.lines = self.file.readlines()
        self.sentences = []
        self.tags = []
        self.histories = []

        #   features statistics
        self.feature_100_counts = OrderedDict()
        self.feature_101_counts = OrderedDict()
        self.feature_102_counts = OrderedDict()
        self.feature_103_counts = OrderedDict()
        self.feature_104_counts = OrderedDict()
        self.feature_105_counts = OrderedDict()

        #   feature2id
        self.feature_100 = OrderedDict()
        self.feature_101 = OrderedDict()
        self.feature_102 = OrderedDict()
        self.feature_103 = OrderedDict()
        self.feature_104 = OrderedDict()
        self.feature_105 = OrderedDict()

        self.prefix_set = set()
        self.suffix_set = set()
        self.words_set = set()
        self.tags_set = set('*')
        # length of feature vector
        self.f_length = -1

    def preprocess(self):
        # Split all the sentences and all the words and tags
        for line in self.lines:
            temp = re.split(' |\n', line)
            del temp[-1]
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
            for word_idx, word in enumerate(sentence):
                word_count += 1
                # history = ( t-2,t-1,t,w )
                t = self.tags[sentence_idx][word_idx]
                t_1 = self.tags[sentence_idx][word_idx-1] if word_idx > 0 else '*'
                t_2 = self.tags[sentence_idx][word_idx-2] if word_idx > 1 else '*'
                curr_history = (t_2, t_1, t, word)
                self.histories.append(curr_history)
                self.words_set.add(word)
                self.tags_set.add(t)
        #   calculate all possible prefixes and suffixes according to the training set
        self.create_prefix_set()
        self.create_suffix_set()
        #   create statistics
        self.calc_features_count()
        #   calculate feature counts
        self.fill_feature_100_dictionary()
        self.fill_feature_101_dictionary()
        self.fill_feature_102_dictionary()
        self.fill_feature_103_dictionary()
        self.fill_feature_104_dictionary()
        self.fill_feature_105_dictionary()

        self.f_length = len(self.feature_100) + len(self.feature_101) + len(self.feature_102) + len(self.feature_103) +\
                        len(self.feature_104) + len(self.feature_105)

    def create_suffix_set(self):
        for word in self.words_set:
            self.suffix_set.add(word[-1:])
            if len(word) >= 2:
                self.suffix_set.add(word[-2:])
                if len(word) >= 3:
                    self.suffix_set.add(word[-3:])
                    if len(word) >= 4:
                        self.suffix_set.add(word[-4:])

    def create_prefix_set(self):
        for word in self.words_set:
            self.prefix_set.add(word[:1])
            if len(word) >= 2:
                self.prefix_set.add(word[:2])
                if len(word) >= 3:
                    self.prefix_set.add(word[:3])
                    if len(word) >= 4:
                        self.prefix_set.add(word[:4])

    def calc_features_count(self):
        for h in self.histories:
            # history = (t-2,t-1,t,w)
            t_2 = h[0]
            t_1 = h[1]
            tag = h[2]
            word = h[3]
            #   f100
            if (word,tag) in self.feature_100_counts:
                self.feature_100_counts[(word, tag)] += 1
            else:
                self.feature_100_counts[(word, tag)] = 1
            #   f101 suffix , tag
            #   run over all possible suffixes of a given word with its tag
            suffixes_tag = []
            suffixes_tag.append((word[-1:],tag))
            if len(word) >= 2:
                suffixes_tag.append((word[-2:],tag))
                if len(word) >= 3:
                    suffixes_tag.append((word[-3:],tag))
                    if len(word) >= 4:
                        suffixes_tag.append((word[-4:],tag))
            for suf_tag in suffixes_tag:
                if suf_tag in self.feature_101_counts:
                    self.feature_101_counts[suf_tag] += 1
                else:
                    self.feature_101_counts[suf_tag] = 1

            #   f102 prefix, tag
            #   run over all possible prefixes of a given word with its tag
            prefixes_tag = []
            prefixes_tag.append((word[:1],tag))
            if len(word) >= 2:
                prefixes_tag.append((word[:2],tag))
                if len(word) >= 3:
                    prefixes_tag.append((word[:3],tag))
                    if len(word) >= 4:
                        prefixes_tag.append((word[:4],tag))
            for pre_tag in prefixes_tag:
                if pre_tag in self.feature_102_counts:
                    self.feature_102_counts[pre_tag] += 1
                else:
                    self.feature_102_counts[pre_tag] = 1

            #   f103
            if (t_2, t_1, tag) in self.feature_103_counts:
                self.feature_103_counts[(t_2, t_1, tag)] += 1
            else:
                self.feature_103_counts[(t_2, t_1, tag)] = 1
            #   f104
            if (t_1, tag) in self.feature_104_counts:
                self.feature_104_counts[(t_1, tag)] += 1
            else:
                self.feature_104_counts[(t_1, tag)] = 1
            #   f105
            if tag in self.feature_105_counts:
                self.feature_105_counts[tag] += 1
            else:
                self.feature_105_counts[tag] = 1

    def fill_feature_100_dictionary(self):
        idx = 0
        for word in self.words_set:
            for tag in self.tags_set:
                if (word, tag) in self.feature_100_counts and self.feature_100_counts[(word, tag)] >= self.thr:
                    self.feature_100[(word, tag)] = idx
                    idx += 1

    def fill_feature_101_dictionary(self):
        idx = 0
        for tag in self.tags_set:
            for suffix in self.suffix_set:
                key = (suffix, tag)
                if key in self.feature_101_counts and self.feature_101_counts[key] >= self.thr:
                    self.feature_101[key] = idx
                    idx += 1

    def fill_feature_102_dictionary(self):
        idx = 0
        for tag in self.tags_set:
            for prefix in self.prefix_set:
                key = (prefix, tag)
                if key in self.feature_102_counts and self.feature_102_counts[key] >= self.thr:
                    self.feature_102[key] = idx
                    idx += 1

    def fill_feature_105_dictionary(self):
        idx = 0
        for tag in self.tags_set:
            if tag in self.feature_105_counts and self.feature_105_counts[tag] >= self.thr:
                self.feature_105[tag] = idx
                idx += 1

    def fill_feature_103_dictionary(self):
        idx = 0
        for first_tag in self.tags_set:
            for sec_tag in self.tags_set:
                for thrd_tag in self.tags_set:
                    key = (first_tag, sec_tag, thrd_tag)
                    if key in self.feature_103_counts and self.feature_103_counts[key] >= self.thr:
                        self.feature_103[key] = idx
                        idx += 1

    def fill_feature_104_dictionary(self):
        idx = 0
        for first_tag in self.tags_set:
            for sec_tag in self.tags_set:
                key = (first_tag, sec_tag)
                if key in self.feature_104_counts and self.feature_104_counts[key] >= self.thr:
                    self.feature_104[key] = idx
                    idx += 1

    def generate_F(self, H):
        """

        :param H: histories dataset
        :return: F which is all feature vectors as a sparse matrix
        """
        row_inds = []
        col_inds = []
        data = []
        for idx,h in enumerate(H):
            places = self.generate_feature_vector(h)
            for p in places:
                row_inds.append(idx)
                col_inds.append(p)
                data.append(1)
        return csr_matrix((data, (row_inds, col_inds)))

    def generate_H_tag(self):
        H_tag = []
        for history in self.histories:
            # history = (t-2,t-1,t,w)
            t_2 = history[0]
            t_1 = history[1]
            tag = history[2]
            word = history[3]
            for possible_tag in self.tags_set:
                possible_history = (t_2, t_1, possible_tag, word)
                H_tag.append(possible_history)
        return H_tag

    def generate_feature_vector(self, history):
        """

        :param history: one history sample
        :return: the history's feature vector
        """
        # TODO remove np shit
        # history = (t-2,t-1,t,w)
        t_2 = history[0]
        t_1 = history[1]
        tag = history[2]
        word = history[3]

        hot_places = []

        #   f100    (word,tag)
        f100 = np.zeros(len(self.feature_100))
        if (word, tag) in self.feature_100:
            hot_places.append(self.feature_100[(word, tag)])
            f100[self.feature_100[(word, tag)]] = 1

        #   f101    (suffix,tag)
        f101 = np.zeros(len(self.feature_101))
        suffixes_tag = []
        suffixes_tag.append((word[-1:], tag))
        if len(word) >= 2:
            suffixes_tag.append((word[-2:], tag))
            if len(word) >= 3:
                suffixes_tag.append((word[-3:], tag))
                if len(word) >= 4:
                    suffixes_tag.append((word[-4:], tag))
        for suf_tag in suffixes_tag:
            if suf_tag in self.feature_101:
                hot_places.append(self.feature_101[suf_tag])
                f101[self.feature_101[suf_tag]] = 1

        #   f102    (prefix,tag)
        f102 = np.zeros(len(self.feature_102))
        prefixes_tag = []
        prefixes_tag.append((word[:1], tag))
        if len(word) >= 2:
            prefixes_tag.append((word[:2], tag))
            if len(word) >= 3:
                prefixes_tag.append((word[:3], tag))
                if len(word) >= 4:
                    prefixes_tag.append((word[:4], tag))
        for pre_tag in prefixes_tag:
            if pre_tag in self.feature_102:
                hot_places.append(self.feature_102[pre_tag])
                f102[self.feature_102[pre_tag]] = 1

        #   f103    (tag,tag,tag)
        f103 = np.zeros(len(self.feature_103))
        if (t_2,t_1,tag) in self.feature_103:
            hot_places.append(self.feature_103[(t_2,t_1,tag)])
            f103[self.feature_103[(t_2,t_1,tag)]] = 1

        #   f104    (tag,tag)
        f104 = np.zeros(len(self.feature_104))
        if (t_1,tag) in self.feature_104:
            hot_places.append(self.feature_104[(t_1,tag)])
            f104[self.feature_104[(t_1,tag)]] = 1

        #   f105    (tag)
        f105 = np.zeros(len(self.feature_105))
        if tag in self.feature_105:
            hot_places.append(self.feature_105[tag])
            f105[self.feature_105[tag]] = 1

        final_feature_vector = np.concatenate((f100, f101, f102, f103, f104, f105))
        #return final_feature_vector
        return hot_places

    # TODO if generate_feature_vector works delete it
    def generate_feature_vector2(self, history):
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
