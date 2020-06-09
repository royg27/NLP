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


class DnnPosTagger(nn.Module):
    def __init__(self, word_embedding_dim, hidden_dim, word_vocab_size, tag_vocab_size):
        super(DnnPosTagger, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        # self.word_embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        self.lstm = nn.LSTM(input_size=word_embedding_dim, hidden_size=hidden_dim, num_layers=2, bidirectional=True,
                            batch_first=False)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tag_vocab_size)

    def forward(self, word_idx_tensor):
        embeds = self.word_embedding(word_idx_tensor.to(self.device))  # [batch_size, seq_length, emb_dim]
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]
        tag_space = self.hidden2tag(lstm_out.view(embeds.shape[1], -1))  # [seq_length, tag_dim]
        tag_scores = F.log_softmax(tag_space, dim=1)  # [seq_length, tag_dim]
        return tag_scores


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

    model = DnnPosTagger(WORD_EMBEDDING_DIM, HIDDEN_DIM, word_vocab_size, tag_vocab_size).to(device)
    if use_cuda:
        model.cuda()

    loss_function = nn.NLLLoss()
    # We will be using a simple SGD optimizer to minimize the loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
            words_idx_tensor.to(device)
            pos_idx_tensor.to(device)
            heads_tensor.to(device)
            model_output = model(words_idx_tensor)

            loss = loss_function(model_output, pos_idx_tensor[0])
            exit()
            loss = loss / accumulate_grad_steps
            batch_loss += loss
            if i % accumulate_grad_steps == 0:
                print("batch done w batch_loss = ", batch_loss)
                optimizer.step()
                optimizer.zero_grad()  # or opt?
                loss_list.append(batch_loss)
                batch_loss = 0

    print("loss_list", loss_list)
    print("accuracy_list", accuracy_list)
    return


main()