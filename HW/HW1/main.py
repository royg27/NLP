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
    processor_thr = [3,5,7]
    model_lambda = [0.1, 0.5, 0.9, 1, 2]
    use_additional_features = [True, False]
    max_val = 0
    best_lmbda = -1
    best_thr = -1
    best_beam = -1
    best_use_additional = True
    s_val = textProcessor(['data/test1.wtag'])
    s_val.preprocess()
    Y = s_val.tags[0:num_sentences]
    for use_extra in use_additional_features:
        for thr in processor_thr:
            for lmbda in model_lambda:
                print("testing thr =",thr, " lambda = ", lmbda)
                s = textProcessor(['data/train1.wtag'],thr=thr, use_extra_features=use_extra)
                s.preprocess()
                model = MEMM(s,lamda=lmbda)
                model.fit(False)
                y_pred = model.predict('data/test1.wtag', num_sentences=num_sentences, beam=3)
                cur_acc = model.accuracy(Y, y_pred)
                if cur_acc > max_val:
                    best_lmbda = lmbda
                    best_thr = thr
                    max_val = cur_acc
                    best_beam = 3
                    best_use_additional = use_extra
                y_pred = model.predict('data/test1.wtag', num_sentences=num_sentences, beam=5)
                cur_acc = model.accuracy(Y, y_pred)
                if cur_acc > max_val:
                    best_lmbda = lmbda
                    best_thr = thr
                    max_val = cur_acc
                    best_beam = 5
                    best_use_additional = use_extra
    print("best acc = ",max_val, " thr = ", best_thr, " lambda = ", best_lmbda, " beam = ",best_beam, " use extra = ", best_use_additional)


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
    # fill_file()
    hyperparameter_tuning()
    # training_part()
    return


if __name__ == "__main__":
    main()