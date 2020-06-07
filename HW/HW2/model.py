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
        self.MLP = nn.Linear(hidden_dim * 4, 1)

    def forward(self, word_idx_tensor, pos_idx_tensor, head_tensor):
        # get x = concat(e(w), e(p))
        e_w = self.word_embedding(word_idx_tensor.to(self.device))  # [batch_size, seq_length, e_w]
        e_p = self.pos_embedding(pos_idx_tensor.to(self.device))    # [batch_size, seq_length, e_p]
        embeds = torch.cat((e_w, e_p), dim=2)                       # [batch_size, seq_length, e_w + e_p]
        assert embeds.shape[0] == 1 and embeds.shape[2] == 105
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]
        #return lstm_out
        # Turns the output into one big tensor, each line is  rep of a word in the sentence
        v_table = lstm_out.view(lstm_out.shape[0], -1)  # [seq_length, tag_dim]
        sum = 0
        #print(head_tensor)
        for idx, head in enumerate(head_tensor[0]):
            if idx==0:
                continue
            first_idx = head.item()
            sec_idx = idx
            # Head always before the modifier
            mlp_input = torch.cat((v_table[first_idx], v_table[sec_idx]), dim=0)
            sum += self.MLP(mlp_input)
        return sum
        # tag_scores = F.log_softmax(tag_space, dim=1)  # [seq_length, tag_dim]
        # return tag_scores


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
    train_dataloader = DataLoader(train, shuffle=True)
    test = PosDataset(word_dict, pos_dict, data_dir, 'test')
    test_dataloader = DataLoader(test, shuffle=False)


    a = next(iter(train_dataloader))
    #a[0] -> word - idx of a sentence
    #a[1] -> pos - idx of a sentence
    #a[2] -> head token per sentence
    assert len(a[0])==len(a[1])==len(a[2])

    EPOCHS = 15
    WORD_EMBEDDING_DIM = 100
    POS_EMBEDDING_DIM = 5
    HIDDEN_DIM = 1000
    word_vocab_size = len(train.word2idx)
    tag_vocab_size = len(train.pos_idx_mappings)

    model = DnnDependencyParser(WORD_EMBEDDING_DIM, POS_EMBEDDING_DIM, HIDDEN_DIM, word_vocab_size, tag_vocab_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

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
        for batch_idx, input_data in enumerate(train_dataloader):
            i += 1
            words_idx_tensor, pos_idx_tensor, heads_tensor = input_data
            lstm_out = model(words_idx_tensor, pos_idx_tensor, heads_tensor)
            print(lstm_out)
            return


main()