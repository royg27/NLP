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
from os import path

# taken from the paper
MLP_HIDDEN_DIM = 100
EPOCHS = 150
WORD_EMBEDDING_DIM = 100
POS_EMBEDDING_DIM = 25
HIDDEN_DIM = 125
LEARNING_RATE = 0.01
EARLY_STOPPING = 10  # num epochs with no validation acc improvement to stop training
PATH = "./basic_model_best_params"

HYPER_PARAMETER_TUNING = True

cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

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


class AdvancedDnnDependencyParser(nn.Module):
    def __init__(self, word_embedding_dim, pos_embedding_dim, hidden_dim, word_vocab_size, tag_vocab_size, num_lst_layers=2):
        super(AdvancedDnnDependencyParser, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # get a tensor of size word_vocab_size and return a word embedding
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        # get a tensor of size tag_vocab_size and return a pos embedding
        self.pos_embedding = nn.Embedding(tag_vocab_size, pos_embedding_dim)
        self.lstm = nn.LSTM(input_size=word_embedding_dim + pos_embedding_dim, hidden_size=hidden_dim, num_layers=num_lst_layers,
                            bidirectional=True, batch_first=False)
        self.mlp = MLP(2*hidden_dim, MLP_HIDDEN_DIM)
        # self.mlp = not_efficientMLP(2*hidden_dim, MLP_HIDDEN_DIM)

    def forward(self, word_idx_tensor, pos_idx_tensor):
        # get x = concat(e(w), e(p))
        e_w = self.word_embedding(word_idx_tensor.to(self.device))                  # [batch_size, seq_length, e_w]
        e_p = self.pos_embedding(pos_idx_tensor.to(self.device))                    # [batch_size, seq_length, e_p]
        embeds = torch.cat((e_w, e_p), dim=2).to(self.device)                       # [batch_size, seq_length, e_w + e_p]
        # assert embeds.shape[0] == 1 and embeds.shape[2] == POS_EMBEDDING_DIM + WORD_EMBEDDING_DIM
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
    return (1.0/sentence_length) * loss


# def NLLL(output, target):
#     """
#     :param output: The table of MLP scores of each word pair
#     :param target: The ground truth of the actual arcs
#     :return:
#     """
#     # loss = -1/|Y|*[S_gt - sum(log(sum(exp(s_j_m))))]
#     S_gt = 0
#     mod_score = 0
#     for idx, head in enumerate(target[0]):
#         if idx == 0:
#             continue
#         head_idx = head.item()
#         mod_idx = idx
#         S_gt += output[head_idx, mod_idx]
#         #
#         S_j_m = output[:, mod_idx]
#         mod_score += torch.log(torch.sum(torch.exp(S_j_m)))
#     Y_i = target[0].shape[0]
#     final_loss = (-1./Y_i)*(S_gt - mod_score)
#     return final_loss



def accuracy(ground_truth, energy_table):
    predicted_mst, _ = decode_mst(energy=energy_table.detach(), length=energy_table.shape[0], has_labels=False)
    # first one is the HEAD of root so we avoid taking it into account
    y_pred = torch.from_numpy(predicted_mst[1:])
    y_true = ground_truth[1:]
    acc = (y_pred == y_true).sum()/float(y_true.shape[0])
    return acc.item()


def evaluate(model, data_loader):
    val_acc = 0
    val_size = 0
    for batch_idx, input_data in enumerate(data_loader):
        val_size += 1
        with torch.no_grad():
            words_idx_tensor, pos_idx_tensor, heads_tensor = input_data
            tag_scores = model(words_idx_tensor, pos_idx_tensor)
            val_acc += (accuracy(heads_tensor[0].cpu(), tag_scores.cpu()))
    return val_acc / val_size


def hyper_parameter_tuning():
    mlp_hidden_dim_arr = [100, 200]
    EPOCHS = 150
    LSTM_LAYERS = [2, 4]
    word_embedding_dim_arr = [100, 200]
    pos_embedding_dim_arr = [25]
    hidden_dim_arr = [125, 175]

    # sanity check
    data_dir = "HW2-files/"
    path_train = data_dir + "train.labeled"
    print("path_train -", path_train)
    path_test = data_dir + "test.labeled"
    print("path_test -", path_test)

    paths_list = [path_train, path_test]
    word_cnt, word_dict, pos_dict = get_vocabs(paths_list)
    train = PosDataset(word_cnt, word_dict, pos_dict, data_dir, 'train')
    # split into validation
    train_set, val_set = torch.utils.data.random_split(train, [4000, 1000])
    train_dataloader = DataLoader(train_set, shuffle=False)  # TODO return to true after debugging
    val_dataloader = DataLoader(val_set, shuffle=False)
    test = PosDataset(word_cnt, word_dict, pos_dict, data_dir, 'test')
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

    max_acc = 0
    max_mlp_hidden_dim = 0
    max_word_embedding_dim = 0
    max_pos_embedding_dim = 0
    max_hidden_dim = 0
    max_learning_rate = 0
    max_lstm_layers = 0

    for mlp_h_d in mlp_hidden_dim_arr:
        for word_e_d in word_embedding_dim_arr:
            for pos_e_d in pos_embedding_dim_arr:
                for hidden in hidden_dim_arr:
                    for num_lstm_layers in LSTM_LAYERS:
                        model = AdvancedDnnDependencyParser(word_e_d, pos_e_d, hidden, word_vocab_size, tag_vocab_size,
                                                            num_lst_layers=num_lstm_layers).to(device)
                        if use_cuda:
                            model.cuda()

                        # Define the loss function as the Negative Log Likelihood loss (NLLLoss)
                        loss_function = nn.NLLLoss()

                        # We will be using a simple SGD optimizer to minimize the loss function
                        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
                        acumulate_grad_steps = 128

                        accuracy_list = []
                        loss_list = []
                        best_val_acc = 0
                        num_epochs_no_improvement = 0
                        for epoch in range(EPOCHS):
                            val_acc = evaluate(model, val_dataloader)
                            if val_acc < best_val_acc:  # no improvement
                                num_epochs_no_improvement += 1
                                if num_epochs_no_improvement >= EARLY_STOPPING:
                                    # best config acc is saved in best_val_acc
                                    print(f"mlp_hidden: {mlp_h_d}, word_emb: {word_e_d}, pos_emb: {pos_e_d}, lstm_hidden: "
                                          f"{hidden}, num_lstm_layers:{num_lstm_layers} -> acc: {val_acc}")
                                    if val_acc > max_acc:
                                        max_acc = val_acc
                                        max_mlp_hidden_dim = mlp_h_d
                                        max_word_embedding_dim = word_e_d
                                        max_pos_embedding_dim = pos_e_d
                                        max_hidden_dim = hidden
                                        max_lstm_layers = num_lstm_layers
                                    break
                            else:  # improvement
                                # torch.save(model.state_dict(), PATH)
                                num_epochs_no_improvement = 0
                                best_val_acc = val_acc

                            # train
                            acc = 0  # to keep track of accuracy
                            printable_loss = 0  # To keep track of the loss value
                            i = 0
                            batch_loss = 0
                            batch_acc = 0
                            for batch_idx, input_data in enumerate(train_dataloader):
                                i += 1
                                words_idx_tensor, pos_idx_tensor, heads_tensor = input_data

                                tag_scores = model(words_idx_tensor, pos_idx_tensor)
                                loss = NLLL_function(tag_scores, heads_tensor[0].to(device))
                                loss = loss / acumulate_grad_steps
                                loss.backward()
                                batch_loss += loss
                                acc = (accuracy(heads_tensor[0].cpu(), tag_scores.cpu())) / acumulate_grad_steps
                                batch_acc += acc
                                if i % acumulate_grad_steps == 0:
                                    optimizer.step()
                                    model.zero_grad()
                                    batch_loss = 0
                                    batch_acc = 0
                                printable_loss += loss.item()
                                _, indices = torch.max(tag_scores, 1)
    print("best params:")
    print(f"mlp_hidden: {max_mlp_hidden_dim}, word_emb: {max_word_embedding_dim}, pos_emb: {max_pos_embedding_dim}, lstm_hidden: "
          f"{max_hidden_dim}, num_lstm_layers:{max_lstm_layers} -> acc: {max_acc}")


def main():
    # sanity check
    data_dir = "HW2-files/"
    path_train = data_dir + "train.labeled"
    print("path_train -", path_train)
    path_test = data_dir + "test.labeled"
    print("path_test -", path_test)

    paths_list = [path_train, path_test]
    word_cnt, word_dict, pos_dict = get_vocabs(paths_list)
    train = PosDataset(word_cnt, word_dict, pos_dict, data_dir, 'train')
    # split into validation
    train_set, val_set = torch.utils.data.random_split(train, [4000, 1000])
    train_dataloader = DataLoader(train_set, shuffle=False)  # TODO return to true after debugging
    val_dataloader = DataLoader(val_set, shuffle=False)
    test = PosDataset(word_cnt, word_dict, pos_dict, data_dir, 'test')
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

    model = AdvancedDnnDependencyParser(WORD_EMBEDDING_DIM, POS_EMBEDDING_DIM, HIDDEN_DIM, word_vocab_size, tag_vocab_size).to(device)
    if use_cuda:
        model.cuda()

    # Define the loss function as the Negative Log Likelihood loss (NLLLoss)
    loss_function = nn.NLLLoss()

    # We will be using a simple SGD optimizer to minimize the loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    acumulate_grad_steps = 128  # This is the actual batch_size, while we officially use batch_size=1

    # Training start
    print("Training Started")
    epoch_loss_list = []
    epoch_train_acc_list = []
    epoch_test_acc_list = []
    best_val_acc = 0
    num_epochs_wo_improvement = 0
    for epoch in range(EPOCHS):
        val_acc = evaluate(model, val_dataloader)
        print("EPOCH = ", epoch)
        print("EPOCH val acc = ", val_acc)
        if val_acc < best_val_acc:     # no improvement
            num_epochs_wo_improvement += 1
            if num_epochs_wo_improvement >= EARLY_STOPPING:
                print("STOPPED TRAINING DUE TO EARLY STOPPING")
                return
        else:                                   # improvement
            print("saving model since it improved on validation :)")
            torch.save(model.state_dict(), PATH)
            num_epochs_wo_improvement = 0
            best_val_acc = val_acc
            fig = plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(epoch_loss_list)
            plt.title("loss")
            plt.subplot(3, 1, 2)
            plt.plot(epoch_train_acc_list)
            plt.title("train UAS")
            plt.subplot(3, 1, 3)
            plt.plot(epoch_test_acc_list)
            plt.title("test UAS")
            print(epoch_train_acc_list)
            plt.savefig('./basic_model_graphs.png')

        # train
        acc = 0  # to keep track of accuracy
        printable_loss = 0  # To keep track of the loss value
        i = 0
        batch_loss = 0
        batch_acc = 0
        epoch_loss = 0

        for batch_idx, input_data in enumerate(train_dataloader):
            i += 1
            words_idx_tensor, pos_idx_tensor, heads_tensor = input_data

            tag_scores = model(words_idx_tensor, pos_idx_tensor)
            loss = NLLL_function(tag_scores, heads_tensor[0].to(device))
            # epoch statistics
            epoch_loss += loss
            #
            loss = loss / acumulate_grad_steps
            loss.backward()
            batch_loss += loss
            acc = (accuracy(heads_tensor[0].cpu(), tag_scores.cpu())) / acumulate_grad_steps
            batch_acc += acc
            if i % acumulate_grad_steps == 0:
                optimizer.step()
                model.zero_grad()
                print("batch_loss = ", batch_loss.item())
                print("batch_acc = ", batch_acc)
                batch_loss = 0
                batch_acc = 0
        # end of epoch - get statistics
        epoch_loss_list.append(epoch_loss / i)
        epoch_train_acc_list.append(evaluate(model, train_dataloader))
        epoch_test_acc_list.append(evaluate(model, test_dataloader))
    # end of train - plot the two graphs
    fig = plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(epoch_loss_list)
    plt.title("loss")
    plt.subplot(3, 1, 2)
    plt.plot(epoch_train_acc_list)
    plt.title("train UAS")
    plt.subplot(3, 1, 3)
    plt.plot(epoch_test_acc_list)
    plt.title("test UAS")
    plt.show()
    plt.savefig('basic_model_graphs.png')


if __name__ == "__main__" :
    if HYPER_PARAMETER_TUNING:
        hyper_parameter_tuning()
    else:
        main()