from collections import defaultdict
# from torchtext.vocab import Vocab
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
from utils import *
import matplotlib.pyplot as plt
from chu_liu_edmonds import *

MLP_HIDDEN_DIM = 100
EPOCHS = 15
WORD_EMBEDDING_DIM = 100
POS_EMBEDDING_DIM = 25
HIDDEN_DIM = 125

cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

class SplittedMLP(nn.Module):
    def __init__(self, lstm_dim, mlp_hidden_dim):
        super(SplittedMLP, self).__init__()
        self.fc_h = nn.Linear(lstm_dim, mlp_hidden_dim, bias=True)  # fully-connected to output mu
        self.fc_m = nn.Linear(lstm_dim, mlp_hidden_dim, bias=False)  # fully-connected to output mu

    def forward(self, lstm_out):
        heads_hidden = self.fc_h(lstm_out)
        mods_hidden = self.fc_m(lstm_out)
        return heads_hidden, mods_hidden


class MLP(nn.Module):
    def __init__(self, lstm_dim, mlp_hidden_dim):
        super(MLP, self).__init__()
        self.first_mlp = SplittedMLP(lstm_dim, mlp_hidden_dim)
        self.non_linearity = nn.ReLU()
        self.second_mlp = nn.Linear(mlp_hidden_dim, 1, bias=True)   # will output a score of a pair

    def forward(self, lstm_out):
        sentence_length = lstm_out.shape[0]
        heads_hidden, mods_hidden = self.first_mlp(lstm_out)
        scores = torch.zeros(size=(sentence_length, sentence_length))
        # we will fill the table row by row, using broadcasting
        for mod in range(sentence_length):
            mod_hidden = mods_hidden[mod]
            summed_values = mod_hidden + heads_hidden   # a single mod with all heads possibilities
            x = self.non_linearity(summed_values)
            scores[:, mod] = torch.flatten(self.second_mlp(x))
        return scores


class DnnDependencyParser(nn.Module):
    def __init__(self, word_embedding_dim, pos_embedding_dim, hidden_dim, word_vocab_size, tag_vocab_size):
        super(DnnDependencyParser, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # get a tensor of size word_vocab_size and return a word embedding
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        # get a tensor of size tag_vocab_size and return a pos embedding
        self.pos_embedding = nn.Embedding(tag_vocab_size, pos_embedding_dim)
        self.lstm = nn.LSTM(input_size=word_embedding_dim + pos_embedding_dim, hidden_size=hidden_dim, num_layers=2,
                            bidirectional=True, batch_first=False)
        self.mlp = MLP(2*hidden_dim, MLP_HIDDEN_DIM)

    # def create_table(self, post_lstm_embedding):
    #     # post_lstm_embedding.shape -> sentence_length, V_size
    #     sentence_length = post_lstm_embedding.shape[0]
    #     table = torch.zeros(size=(sentence_length, sentence_length))
    #     mlp_heads = []
    #     mlp_mods = []
    #     for i in range(sentence_length):
    #         mlp_heads.append(self.MLP_head(post_lstm_embedding[i]).item())
    #         mlp_mods.append(self.MLP_modifier(post_lstm_embedding[i]).item())
    #     # construct table
    #     for h in range(sentence_length):
    #         for m in range(sentence_length):
    #             table[h, m] = mlp_heads[h] + mlp_mods[m]
    #     return table

    def forward(self, word_idx_tensor, pos_idx_tensor, head_tensor):
        # get x = concat(e(w), e(p))
        e_w = self.word_embedding(word_idx_tensor.to(self.device))  # [batch_size, seq_length, e_w]
        e_p = self.pos_embedding(pos_idx_tensor.to(self.device))    # [batch_size, seq_length, e_p]
        embeds = torch.cat((e_w, e_p), dim=2)                       # [batch_size, seq_length, e_w + e_p]
        assert embeds.shape[0] == 1 and embeds.shape[2] == POS_EMBEDDING_DIM + WORD_EMBEDDING_DIM
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]
        # Turns the output into one big tensor, each line is  rep of a word in the sentence
        lstm_out = lstm_out.view(lstm_out.shape[0], -1)  # [seq_length, 2*hidden_dim]
        out = self.mlp(lstm_out)
        return out

def NLLL_function(scores, true_tree):
    """
    Parameters
    ----------
    scores - a matrix of size (sentence_length x sentence length)
    true_tree - ground truth dependency tree

    Returns the loss
    -------
    """
    clean_scores = scores[:, 1:]                # ROOT cant be modifier
    clean_true_tree = true_tree[1:]
    sentence_length = clean_scores.shape[1]     # without root
    loss = 0
    for mod in range(sentence_length):
        cross_entropy_loss(scores[:, mod].unsqueeze(dim=0), clean_true_tree[mod:mod+1])
        loss += cross_entropy_loss(scores[:,mod], clean_true_tree[mod])
    return loss


def NLLL(output, target):
    """
    :param output: The table of MLP scores of each word pair
    :param target: The ground truth of the actual arcs
    :return:
    """
    # loss = -1/|Y|*[S_gt - sum(log(sum(exp(s_j_m))))]
    S_gt = 0
    mod_score = 0
    for idx, head in enumerate(target[0]):
        if idx == 0:
            continue
        head_idx = head.item()
        mod_idx = idx
        S_gt += output[head_idx, mod_idx]
        #
        S_j_m = output[:, mod_idx]
        mod_score += torch.log(torch.sum(torch.exp(S_j_m)))
    Y_i = target[0].shape[0]
    final_loss = (-1./Y_i)*(S_gt - mod_score)
    return final_loss


def get_acc_measurements(GT, energy_table):
    predicted_mst, _ = decode_mst(energy=energy_table, length=energy_table.shape[0], has_labels=False)
    y_pred = torch.from_numpy(predicted_mst[1:])
    y_true = GT[1:]
    print("y_pred", y_pred)
    print("y_true = ", y_true)
    print((y_pred == y_true).sum())
    acc = (y_pred == y_true).sum()/float(y_true.shape[0])
    return acc.item()


def accuracy(ground_truth, energy_table):
    predicted_mst, _ = decode_mst(energy=energy_table.detach(), length=energy_table.shape[0], has_labels=False)
    # first one is the HEAD of root so we avoid taking it into account
    y_pred = torch.from_numpy(predicted_mst[1:])
    y_true = ground_truth[1:]
    acc = (y_pred == y_true).sum()/float(y_true.shape[0])
    return acc.item()


def main():
    # sanity check
    data_dir = "HW2-files/"
    path_train = data_dir + "train.labeled"
    print("path_train -", path_train)
    path_test = data_dir + "test.labeled"
    print("path_test -", path_test)

    paths_list = [path_train, path_test]
    word_dict, pos_dict = get_vocabs(paths_list)
    train = PosDataset(word_dict, pos_dict, data_dir, 'train')
    train_dataloader = DataLoader(train, shuffle=False) # TODO return to true after debugging
    test = PosDataset(word_dict, pos_dict, data_dir, 'test')
    test_dataloader = DataLoader(test, shuffle=False)


    a = next(iter(train_dataloader))
    #a[0] -> word - idx of a sentence
    #a[1] -> pos - idx of a sentence
    #a[2] -> head token per sentence
    assert len(a[0])==len(a[1])==len(a[2])


    word_vocab_size = len(train.word2idx)
    tag_vocab_size = len(train.pos_idx_mappings)

    model = DnnDependencyParser(WORD_EMBEDDING_DIM, POS_EMBEDDING_DIM, HIDDEN_DIM, word_vocab_size, tag_vocab_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        model.cuda()


    # We will be using a simple SGD optimizer to minimize the loss function
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    accumulate_grad_steps = 50  # This is the actual batch_size, while we officially use batch_size=1

    # Training start
    print("Training Started")
    accuracy_list = []
    loss_list = []
    epochs = EPOCHS
    for epoch in range(epochs):
        print("EPOCH = ", epoch)
        acc = 0  # to keep track of accuracy
        printable_loss = 0  # To keep track of the loss value
        i = 0
        batch_loss = 0
        batch_acc = 0
        for batch_idx, input_data in enumerate(train_dataloader):
            i += 1
            words_idx_tensor, pos_idx_tensor, heads_tensor = input_data
            model_output = model(words_idx_tensor, pos_idx_tensor, heads_tensor)
            loss = NLLL_function(model_output, heads_tensor[0])
            loss = loss / accumulate_grad_steps
            batch_loss += loss
            acc = accuracy(ground_truth=heads_tensor[0], energy_table=model_output)
            batch_acc += acc

            if i % accumulate_grad_steps == 0:
                print("batch done w acc = ", (1.0*batch_acc) / accumulate_grad_steps)
                optimizer.step()
                optimizer.zero_grad()
                loss_list.append(batch_loss)
                accuracy_list.append(batch_acc / accumulate_grad_steps)
                batch_acc = 0
                batch_loss = 0

    print("loss_list", loss_list)
    print("accuracy_list", accuracy_list)
    return


main()