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


def hyperparameter_tuning():
    # best found - thr = 5, lambda = 1 -> 83.67%
    processor_thr = [5]
    model_lambda = [0.9, 1,2]
    max_val = 0
    best_lmbda = -1
    best_thr = -1
    for thr in processor_thr:
        for lmbda in model_lambda:
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

def main():
    # train processor
    # s = train_processor()
    # train model
    # model = train_model(s)
    # unit_tests.check_text_processor_basic()
    # unit_tests.test_model(1)
    # # predict
    #s = textProcessor(['data/train1.wtag','data/train2.wtag'],thr=5)
    s = textProcessor(['data/train1.wtag'], thr=5)
    s.preprocess()
    model = MEMM(s, lamda=1)
    model.fit(True)
    print("predicting")
    Y_pred = model.predict('data/test1.wtag', num_sentences=3, beam=3)
    print("done predicting")
    s = textProcessor(['data/test1.wtag'], thr=5)
    s.preprocess()
    Y = s.tags
    acc = model.accuracy(Y[:3], Y_pred)
    model.confusion_matrix_roy(Y[:3], Y_pred)
    print("acc = ", acc)
    # # hyperparameter search
    # hyperparameter_tuning()
    return


if __name__ == "__main__":
    main()