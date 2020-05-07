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
    init_v = np.copy(model.v)

    start = time.time()
    model.fit(False)
    end = time.time()
    print("training time = ",end-start)
    return model

def fill_file():
    train_file = 'data/train1.wtag'
    s = textProcessor(['data/train2.wtag'], thr=5)
    s.preprocess()
    model = MEMM(s, lamda=1, sigma=0.001)
    model.fit(False)
    predict_file = 'data/comp2.words'
    model.add_prediction_to_file_roy(predict_file, num_sentences=-1, beam=3)

def hyperparameter_tuning():
    # best found - thr = 5, lambda = 1 -> 83.67%
    processor_thr = [5]
    model_lambda = [0.9, 1,2]
    max_val = 0
    best_lmbda = -1
    best_thr = -1
    for thr in processor_thr:
        for lmbda in model_lambda:
            thr = 5
            lmbda = 1
            print("testing thr =",thr, " lambda = ", lmbda)
            s = textProcessor(['data/train1.wtag'],thr=thr)
            s.preprocess()
            model = MEMM(s,lamda=lmbda)
            model.fit(False)
            val = model.predict('data/train2.wtag', verbose=False, num_sentences=100, beam=3)
            if val > max_val:
                best_lmbda = lmbda
                best_thr = thr
                max_val = val
    print("best acc = ",max_val, " thr = ", best_thr, " lambda = ", best_lmbda)

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

def roy():
    # train processor
    # s = train_processor()
    # train model
    # model = train_model(s)
    # unit_tests.check_text_processor_basic()
    # unit_tests.test_model(1)
    # # predict
    # s = textProcessor(['data/train1.wtag','data/train2.wtag'],thr=5)
    model = MEMM(s, lamda=1)
    model.fit(False)
    print("predicting")
    num_sentences_to_tag = 100
    Y_pred = model.predict('data/test1.wtag', num_sentences=num_sentences_to_tag, beam=3)
    print("done predicting")
    s = textProcessor(['data/test1.wtag'], thr=5)
    s.preprocess()
    Y = s.tags
    acc = model.accuracy(Y[:num_sentences_to_tag], Y_pred)
    model.confusion_matrix_roy(Y[:num_sentences_to_tag], Y_pred)
    print("acc = ", acc)
    # # hyperparameter search
    # hyperparameter_tuning()
    #fill_file()
    return

def main():
    training_part()
    return


if __name__ == "__main__":
    main()