from textProcessor import textProcessor
from model import MEMM
import settings
import time
import numpy as np

def check_text_processor_basic():
    #full of assert, should just finish running
    s = textProcessor('data/train1.wtag')
    s.preprocess()
    H = s.histories
    h = H[0][0]
    f = s.generate_feature_vector(h)
    fs = s.generate_expected_count_features(h)
    print(f.shape)
    print(fs.dot(f))

def main():
    #check_text_processor_basic()
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