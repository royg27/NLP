from textProcessor import textProcessor
from model import MEMM
import time
import numpy as np
from numpy import loadtxt
from numpy.random import randint


def check_text_processor_basic():
    #full of assert, should just finish running
    s = textProcessor('data/train1.wtag')
    start = time.time()
    s.preprocess()
    end = time.time()
    print("analyzing dictionaries")
    f100 = s.feature_100
    f101 = s.feature_101
    f102 = s.feature_102
    f103 = s.feature_103
    f104 = s.feature_104
    f105 = s.feature_105
    assert np.all(f100.values() != f101.values())
    assert np.all(f102.values() != f103.values())
    assert np.all(f104.values() != f104.values())
    assert np.all(f104.values() != f105.values())

    print("analyzing dictionaries - Done")
    print("preprocess took ", end - start)
    H = s.histories
    h = H[0]
    H_tag = s.generate_H_tag()
    h = H[0]
    start = time.time()
    F=s.generate_F(H)
    end = time.time()
    print("calc_F took ", end - start)
    start = time.time()
    print(s.generate_F(H_tag))
    end = time.time()
    print("calc_F' took ", end - start)

def test_model(iters = 10,file='data/train1.wtag'):
    print("running unit test on our model")
    s = textProcessor(file)
    s.preprocess()
    trained_weights = loadtxt('weights.csv')
    H = s.histories
    acc = 0
    for i in range(iters):
        #idx = randint(0, len(H))
        idx = i
        h = H[idx]
        H_tag = s.generate_H_tag()
        F_tag = s.generate_F(H_tag[idx*len(s.tags_set):(idx+1)*len(s.tags_set)])
        a=F_tag.todense()
        model = MEMM(s)
        p = model.vectorized_softmax(F_tag,trained_weights, len(s.tags_set))
        t = np.max(p)
        argsort = np.argsort(p)
        tags = list(s.tags_set)
        tags = np.array(tags)
        #
        print("h = ", h,"idx = ",idx, " predicted ", tags[argsort[-4:]], p[argsort[-4:]]," while ground truth is ", h[2])
        acc += int(h[2] in tags[argsort[-3:]])

    print("model's accuracy is ",acc/iters)
