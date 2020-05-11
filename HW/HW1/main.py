from textProcessor import textProcessor
from model import MEMM
import time
import numpy as np
import unit_tests


def train_processor():
    s = textProcessor(['data/train1.wtag'])
    s.preprocess()
    return s


def train_model(s : textProcessor):
    model = MEMM(s)
    start = time.time()
    model.fit(False)
    end = time.time()
    print("training time = ",end-start)
    return model


def fill_file(file_to_predict, tagged_file_name):
    s = textProcessor(['data/train1.wtag'], thr=5)
    s.preprocess()
    model = MEMM(s, lamda=1, sigma=0.001)
    model.fit(False)
    predict_file = file_to_predict
    model.generate_predicted_file(predict_file, tagged_file_name, beam=3)


def hyperparameter_tuning():
    num_sentences = 100
    processor_thr = [3]
    model_lambda = [0.1,0.5,2,10]
    max_val = 0
    best_lmbda = -1
    best_thr = -1
    s_val = textProcessor(['data/test1.wtag'])
    s_val.preprocess()
    Y = s_val.tags[0:num_sentences]
    for thr in processor_thr:
        for lmbda in model_lambda:
            print("testing thr =",thr, " lambda = ", lmbda)
            s = textProcessor(['data/train1.wtag'],thr=thr)
            s.preprocess()
            model = MEMM(s,lamda=lmbda)
            t = time.time()
            model.fit(False)
            end = time.time()
            print("fit time = ", (end - t))
            t = time.time()
            y_pred = model.predict('data/test1.wtag', num_sentences=num_sentences, beam=3)
            end = time.time()
            print("prediction time = ", (end-t))
            cur_acc = model.accuracy(Y, y_pred)
            print("cur_acc = ", cur_acc)
            if cur_acc > max_val:
                best_lmbda = lmbda
                best_thr = thr
                max_val = cur_acc
    print("best acc = ",max_val, " thr = ", best_thr, " lambda = ", best_lmbda, " use extra = ")


def training_part():
    print("model 1 training:")
    s_1 = textProcessor(['data/train1.wtag'], thr=5)
    s_1.preprocess()
    train_model(s_1)

    print("model 2 training:")
    s_2 = textProcessor(['data/train2.wtag'], thr=5)
    s_2.preprocess()
    train_model(s_2)
    return


def generate_confusion_matrix():
    s = textProcessor(['data/train1.wtag'], thr=3)
    s.preprocess()
    num_sentences_to_tag = 1000
    model = MEMM(s, lamda=2)
    model.fit(True)
    Y_pred = model.predict('data/test1.wtag', num_sentences=num_sentences_to_tag, beam=3)
    s_val = textProcessor(['data/test1.wtag'], thr=3)
    s_val.preprocess()
    Y = s_val.tags[:num_sentences_to_tag]
    model.confusion_matrix_roy(Y, Y_pred)


def main():
    # fill_file()
    hyperparameter_tuning()
    # training_part()
    # generate_confusion_matrix()
    return


if __name__ == "__main__":
    main()