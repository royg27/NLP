import os
import numpy as np
import re
from scipy.sparse import csr_matrix
import settings
from collections import OrderedDict


class textProcessor:
    def __init__(self, path_file_name, thr=3, thr_2=3):
        self.thr = thr
        self.thr_common = thr_2
        self.lines = []
        for file in path_file_name:
            self.file = open(file, 'r')
            self.lines = np.concatenate((self.lines, self.file.readlines()))
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
        #   additional features
        self.num_additional = 3
        self.feature_start_cap_counts = OrderedDict()
        self.feature_all_caps_counts = OrderedDict()
        self.feature_contains_numbers_counts = OrderedDict()
        self.feature_contains_hyphen_counts = OrderedDict()
        self.feature_106_counts = OrderedDict()
        self.feature_107_counts = OrderedDict()
        self.feature_start_cap = OrderedDict()
        self.feature_all_caps = OrderedDict()
        self.feature_contains_numbers = OrderedDict()
        self.feature_contains_hyphen = OrderedDict()
        self.feature_106 = OrderedDict()
        self.feature_107 = OrderedDict()
        #
        self.prefix_set = set()
        self.suffix_set = set()
        self.suffix_arr = []
        self.prefix_arr = []
        self.words_set_t = set()
        self.words_set = []
        self.tags_set_t = set('*')
        self.tags_set = []
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
                prev_word = sentence[word_idx-1] if word_idx>0 else '*'
                next_word = sentence[word_idx+1] if word_idx+1<len(sentence) else '*'
                curr_history = (t_2, t_1, t, word, prev_word,next_word)
                self.histories.append(curr_history)
                self.words_set_t.add(word)
                self.tags_set_t.add(t)
        #   calculate all possible prefixes and suffixes according to the training set
        tag_list = list(self.tags_set_t)
        self.tags_set = np.array(tag_list)
        self.tags_set = np.sort(self.tags_set)
        word_list = list(self.words_set_t)
        self.words_set = np.array(word_list)
        self.words_set = np.sort(self.words_set)

        self.create_prefix_set()
        self.create_suffix_set()
        #   create statistics
        self.calc_features_count()
        #   calculate feature counts
        finish_idx = self.fill_feature_100_dictionary()
        finish_idx = self.fill_feature_101_dictionary(finish_idx)
        finish_idx = self.fill_feature_102_dictionary(finish_idx)
        finish_idx = self.fill_feature_103_dictionary(finish_idx)
        finish_idx = self.fill_feature_104_dictionary(finish_idx)
        finish_idx = self.fill_feature_105_dictionary(finish_idx)
        #   additional features
        finish_idx = self.fill_feature_start_cap(finish_idx)
        finish_idx = self.fill_feature_all_caps(finish_idx)
        finish_idx = self.fill_feature_contains_numbers(finish_idx)
        finish_idx = self.fill_feature_contains_hyphen(finish_idx)
        finish_idx = self.fill_feature_106_dictionary(finish_idx)
        # finish_idx = self.fill_feature_107_dictionary(finish_idx)

        # print(len(self.feature_100), len(self.feature_101), len(self.feature_102), len(self.feature_103), len(self.feature_104),
        #       len(self.feature_105), len(self.feature_106), len(self.feature_contains_hyphen), len(self.feature_all_caps),
        #       len(self.feature_contains_numbers), len(self.feature_start_cap))
        #   length of feature
        self.f_length = finish_idx

    def create_suffix_set(self):
        for word in self.words_set:
            self.suffix_set.add(word[-1:])
            if len(word) >= 2:
                self.suffix_set.add(word[-2:])
                if len(word) >= 3:
                    self.suffix_set.add(word[-3:])
                    if len(word) >= 4:
                        self.suffix_set.add(word[-4:])
        self.suffix_arr = list(self.suffix_set)
        self.suffix_arr = np.array(self.suffix_arr)
        self.suffix_arr = np.sort(self.suffix_arr)

    def create_prefix_set(self):
        for word in self.words_set:
            self.prefix_set.add(word[:1])
            if len(word) >= 2:
                self.prefix_set.add(word[:2])
                if len(word) >= 3:
                    self.prefix_set.add(word[:3])
                    if len(word) >= 4:
                        self.prefix_set.add(word[:4])
        self.prefix_arr = list(self.prefix_set)
        self.prefix_arr = np.array(self.prefix_arr)
        self.prefix_arr = np.sort(self.prefix_arr)

    def calc_features_count(self):
        for h in self.histories:
            # history = (t-2,t-1,t,w,prev_word)
            t_2 = h[0]
            t_1 = h[1]
            tag = h[2]
            word = h[3]
            prev_word = h[4]
            next_word = h[5]
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

            #   f106
            if (prev_word, tag) in self.feature_106_counts:
                self.feature_106_counts[(prev_word, tag)] += 1
            else:
                self.feature_106_counts[(prev_word, tag)] = 1
            # #   f107
            # if (next_word, tag) in self.feature_107_counts:
            #     self.feature_107_counts[(next_word, tag)] += 1
            # else:
            #     self.feature_107_counts[(next_word, tag)] = 1

            #   feature_start_cap_counts
            if word[0].isupper() and tag in self.feature_start_cap_counts:
                self.feature_start_cap_counts[tag] += 1
            else:
                self.feature_start_cap_counts[tag] = 1
            #   feature_all_caps
            if word.isupper() and tag in self.feature_all_caps_counts:
                self.feature_all_caps_counts[tag] += 1
            else:
                self.feature_all_caps_counts[tag] = 1
            #   feature_contains_numbers
            if (not word.isalpha()) and tag in self.feature_contains_numbers_counts:
                self.feature_contains_numbers_counts[tag] += 1
            else:
                self.feature_contains_numbers_counts[tag] = 1
            #   contains hyphen
            if '-' in word and tag in self.feature_contains_hyphen_counts:
                self.feature_contains_hyphen_counts[tag] += 1
            else:
                self.feature_contains_hyphen_counts[tag] = 1

    def fill_feature_100_dictionary(self):
        idx = 0
        for word in self.words_set:
            for tag in self.tags_set:
                if (word, tag) in self.feature_100_counts and self.feature_100_counts[(word, tag)] >= self.thr_common:
                    self.feature_100[(word, tag)] = idx
                    idx += 1
        return idx

    def fill_feature_101_dictionary(self, start_idx):
        idx = start_idx
        for tag in self.tags_set:
            for suffix in self.suffix_arr:
                key = (suffix, tag)
                if key in self.feature_101_counts and self.feature_101_counts[key] >= self.thr:
                    self.feature_101[key] = idx
                    idx += 1
        return idx

    def fill_feature_102_dictionary(self, start_idx):
        idx = start_idx
        for tag in self.tags_set:
            for prefix in self.prefix_arr:
                key = (prefix, tag)
                if key in self.feature_102_counts and self.feature_102_counts[key] >= self.thr:
                    self.feature_102[key] = idx
                    idx += 1
        return idx

    def fill_feature_103_dictionary(self, start_idx):
        idx = start_idx
        for first_tag in self.tags_set:
            for sec_tag in self.tags_set:
                for thrd_tag in self.tags_set:
                    key = (first_tag, sec_tag, thrd_tag)
                    if key in self.feature_103_counts and self.feature_103_counts[key] >= self.thr_common:
                        self.feature_103[key] = idx
                        idx += 1
        return idx

    def fill_feature_104_dictionary(self, start_idx):
        idx = start_idx
        for first_tag in self.tags_set:
            for sec_tag in self.tags_set:
                key = (first_tag, sec_tag)
                if key in self.feature_104_counts and self.feature_104_counts[key] >= self.thr_common:
                    self.feature_104[key] = idx
                    idx += 1
        return idx

    def fill_feature_105_dictionary(self, start_idx):
        idx = start_idx
        for tag in self.tags_set:
            if tag in self.feature_105_counts and self.feature_105_counts[tag] >= self.thr_common:
                self.feature_105[tag] = idx
                idx += 1
        return idx

    def fill_feature_start_cap(self,start_idx):
        idx = start_idx
        for tag in self.tags_set:
            if tag in self.feature_start_cap_counts and self.feature_start_cap_counts[tag] >= self.thr:
                self.feature_start_cap[tag] = idx
                idx += 1
        return idx

    def fill_feature_all_caps(self,start_idx):
        idx = start_idx
        for tag in self.tags_set:
            if tag in self.feature_all_caps_counts and self.feature_all_caps_counts[tag] >= self.thr:
                self.feature_all_caps[tag] = idx
                idx += 1
        return idx

    def fill_feature_contains_numbers(self,start_idx):
        idx = start_idx
        for tag in self.tags_set:
            if tag in self.feature_contains_numbers_counts and self.feature_contains_numbers_counts[tag] >= self.thr:
                self.feature_contains_numbers[tag] = idx
                idx += 1
        return idx

    def fill_feature_contains_hyphen(self, start_idx):
        idx = start_idx
        for tag in self.tags_set:
            if tag in self.feature_contains_hyphen_counts and self.feature_contains_hyphen_counts[tag] >= self.thr:
                self.feature_contains_hyphen[tag] = idx
                idx += 1
        return idx

    def fill_feature_106_dictionary(self, start_idx):
        idx = start_idx
        for word in self.words_set:
            for tag in self.tags_set:
                if (word, tag) in self.feature_106_counts and self.feature_106_counts[(word, tag)] >= self.thr_common:
                    self.feature_106[(word, tag)] = idx
                    idx += 1
        return idx

    def fill_feature_107_dictionary(self, start_idx):
        idx = start_idx
        for word in self.words_set:
            for tag in self.tags_set:
                if (word, tag) in self.feature_107_counts and self.feature_107_counts[(word, tag)] >= self.thr:
                    self.feature_107[(word, tag)] = idx
                    idx += 1
        return idx

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
        return csr_matrix((data, (row_inds, col_inds)), shape=(len(H),self.f_length))

    def generate_H_tag(self):
        H_tag = []
        for history in self.histories:
            # history = (t-2,t-1,t,w)
            t_2 = history[0]
            t_1 = history[1]
            tag = history[2]
            word = history[3]
            prev_word = history[4]
            next_word = history[5]
            for possible_tag in self.tags_set:
                possible_history = (t_2, t_1, possible_tag, word, prev_word, next_word)
                H_tag.append(possible_history)
        return H_tag

    def generate_feature_vector(self, history):
        """
        :param history: one history sample
        :return: the history's feature vector
        """
        # history = (t-2,t-1,t,w,prev_word)
        t_2 = history[0]
        t_1 = history[1]
        tag = history[2]
        word = history[3]
        prev_word = history[4]
        next_word = history[5]
        hot_places = []
        #   f100    (word,tag)
        #print('word: ', word, " tag: ", tag)
        if (word, tag) in self.feature_100:
            hot_places.append(self.feature_100[(word, tag)])

        #   f101    (suffix,tag)
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

        #   f102    (prefix,tag)
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

        #   f103    (tag,tag,tag)
        if (t_2,t_1,tag) in self.feature_103:
            hot_places.append(self.feature_103[(t_2,t_1,tag)])

        #   f104    (tag,tag)
        if (t_1,tag) in self.feature_104:
            hot_places.append(self.feature_104[(t_1,tag)])

        #   f105    (tag)
        if tag in self.feature_105:
            hot_places.append(self.feature_105[tag])

        #   start with capital
        if len(word) > 0 and word[0].isupper() and tag in self.feature_start_cap:
            hot_places.append(self.feature_start_cap[tag])

        #   all capital
        if word.isupper() and tag in self.feature_all_caps:
            hot_places.append(self.feature_all_caps[tag])

        #   contains numbers
        if (not word.isalpha()) and (tag in self.feature_contains_numbers):
            hot_places.append(self.feature_contains_numbers[tag])

        #   contains hyphen
        if '_' in word and tag in self.feature_contains_hyphen:
            hot_places.append(self.feature_contains_hyphen[tag])

        #   f106
        if (prev_word, tag) in self.feature_106:
            hot_places.append(self.feature_106[(prev_word, tag)])

        # #   f107
        # if (next_word, tag) in self.feature_107:
        #     hot_places.append(self.feature_107[(next_word, tag)])

        return hot_places

    def generate_h_tag_for_word_roy(self, word,t_2,t_1,prev_word,next_word):
        h_tag = []
        # history = (t-2,t-1,t,w,prev_word,next_word)
        for t in self.tags_set:
            history = (t_2, t_1, t, word, prev_word,next_word)
            h_tag.append(history)
        return h_tag
