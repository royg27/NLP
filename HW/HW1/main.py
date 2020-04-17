from textProcessor import textProcessor
from model import MEMM
import settings
import time
import numpy as np

def check_text_processor_basic():
    #full of assert, should just finish running
    s = textProcessor('data/train1.wtag')
    start = time.time()
    s.preprocess()
    end = time.time()
    print("preprocess took ", end - start)
    H = s.histories
    H_tag = s.generate_H_tag()
    print(len(H))
    print(len(H_tag))
    h = H[0]
    start = time.time()
    print(s.generate_F(H))
    end = time.time()
    print("calc_F took ", end - start)
    start = time.time()
    print(s.generate_F(H_tag))
    end = time.time()
    print("calc_F' took ", end - start)



def main():
    check_text_processor_basic()
    return
    #settings.dont_use_vectorized() 3 vectorized is 3.5 times faster!
    settings.use_vectorized()
    s = textProcessor('data/train1.wtag')
    s.preprocess()
    model = MEMM(s)
    init_v = np.copy(model.v)

    start = time.time()
    model.fit(1)
    end = time.time()

    trained_v = model.v
    comparison = init_v != trained_v
    print(comparison)
    are_different = comparison.all()
    assert are_different
    print("non-vectorized 1 sentence: ",end - start)

    return


if __name__ == "__main__":
    main()