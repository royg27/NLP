from textProcessor import textProcessor
from model import MEMM
import time
import numpy as np
import unit_tests


def train_processor():
    s = textProcessor('data/train1.wtag')
    s.preprocess()
    return s


def train_model(s : textProcessor):
    model = MEMM(s)
    init_v = np.copy(model.v)

    start = time.time()
    model.fit()
    end = time.time()
    return model


def main():

    # train processor
    s = train_processor()
    # train model
    model = train_model(s)
    # unit_tests.check_text_processor_basic()
    unit_tests.test_model(100)
    return


if __name__ == "__main__":
    main()