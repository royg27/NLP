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

MLP_HIDDEN_DIM = 50
EPOCHS = 15
WORD_EMBEDDING_DIM = 100
POS_EMBEDDING_DIM = 25
HIDDEN_DIM = 125
LEARNING_RATE = 0.01

cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class not_efficientMLP(nn.Module):
    def __init__(self, lstm_dim, mlp_hidden_dim):
        super(not_efficientMLP, self).__init__()
        self.first_linear = nn.Linear(2 * lstm_dim, mlp_hidden_dim)
        self.non_linearity = nn.ReLU()
        self.second_mlp = nn.Linear(mlp_hidden_dim, 1, bias=True)  # will output a score of a pair

    def forward(self, lstm_out):
        sentence_length = lstm_out.shape[0]
        scores = torch.zeros(size=(sentence_length, sentence_length)).to(device)
        for i, v_i in enumerate(lstm_out):
            for j, v_j in enumerate(lstm_out):
                if i == j:
                    scores[i, j] = 0
                else:
                    a = torch.cat((v_i, v_j), dim=0)
                    x = self.first_linear(a)
                    y = self.non_linearity(x)
                    scores[i, j] = self.second_mlp(y)
        return scores

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
        scores = torch.zeros(size=(sentence_length, sentence_length)).to(device)
        # we will fill the table row by row, using broadcasting
        for mod in range(sentence_length):
            mod_hidden = mods_hidden[mod]
            summed_values = mod_hidden + heads_hidden   # a single mod with all heads possibilities
            x = self.non_linearity(summed_values)
            scores[:, mod] = torch.flatten(self.second_mlp(x))
            scores[mod, mod] = -np.inf    # a word cant be its head
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
        # self.mlp = not_efficientMLP(2*hidden_dim, MLP_HIDDEN_DIM)

    def forward(self, word_idx_tensor, pos_idx_tensor):
        # get x = concat(e(w), e(p))
        e_w = self.word_embedding(word_idx_tensor.to(self.device))                  # [batch_size, seq_length, e_w]
        e_p = self.pos_embedding(pos_idx_tensor.to(self.device))                    # [batch_size, seq_length, e_p]
        embeds = torch.cat((e_w, e_p), dim=2).to(self.device)                       # [batch_size, seq_length, e_w + e_p]
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
        loss += cross_entropy_loss(clean_scores[:, mod].unsqueeze(dim=0), clean_true_tree[mod:mod+1])
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
    print(y_pred)
    print(y_true)
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
    train_dataloader = DataLoader(train, shuffle=False)  # TODO return to true after debugging
    test = PosDataset(word_dict, pos_dict, data_dir, 'test')
    test_dataloader = DataLoader(test, shuffle=False)

    a = next(iter(train_dataloader))
    # a[0] -> word - idx of a sentence
    # a[1] -> pos - idx of a sentence
    # a[2] -> head token per sentence
    assert len(a[0]) == len(a[1]) == len(a[2])

    word_vocab_size = len(train.word2idx)
    print(word_vocab_size)
    tag_vocab_size = len(train.pos_idx_mappings)
    print(tag_vocab_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = DnnDependencyParser(WORD_EMBEDDING_DIM, POS_EMBEDDING_DIM, HIDDEN_DIM, word_vocab_size, tag_vocab_size).to(device)

    if use_cuda:
        model.cuda()

    # Define the loss function as the Negative Log Likelihood loss (NLLLoss)
    loss_function = nn.NLLLoss()

    # We will be using a simple SGD optimizer to minimize the loss function
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    acumulate_grad_steps = 50  # This is the actual batch_size, while we officially use batch_size=1

    # Training start
    print("Training Started")
    accuracy_list = []
    loss_list = []
    epochs = EPOCHS
    for epoch in range(epochs):
        acc = 0  # to keep track of accuracy
        printable_loss = 0  # To keep track of the loss value
        i = 0
        batch_loss = 0
        for batch_idx, input_data in enumerate(train_dataloader):
            i += 1
            words_idx_tensor, pos_idx_tensor, heads_tensor = input_data

            tag_scores = model(words_idx_tensor, pos_idx_tensor)
            print(tag_scores.shape)
            # print("tag_scores shape -", tag_scores.shape)
            # print("pos_idx_tensor shape -", pos_idx_tensor.shape)
            loss = NLLL_function(tag_scores, heads_tensor[0].to(device))
            loss = loss / acumulate_grad_steps
            loss.backward()
            batch_loss += loss
            if i % acumulate_grad_steps == 0:
                optimizer.step()
                model.zero_grad()
                print(batch_loss)
                batch_loss = 0
            printable_loss += loss.item()
            _, indices = torch.max(tag_scores, 1)
            # print("tag_scores shape-", tag_scores.shape)
            # print("indices shape-", indices.shape)
            # acc += indices.eq(pos_idx_tensor.view_as(indices)).mean().item()
            # acc += torch.mean(torch.tensor(pos_idx_tensor.to("cpu") == indices.to("cpu"), dtype=torch.float))
        # printable_loss = printable_loss / len(train)
        # acc = acc / len(train)
        # loss_list.append(float(printable_loss))
        # accuracy_list.append(float(acc))
        # test_acc = evaluate()
        e_interval = i


main()