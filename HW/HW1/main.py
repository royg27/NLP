from textProcessor import textProcessor
from model import MEMM


def check_text_processor_basic():
    #full of assert, should just finish running
    s = textProcessor('data/train1.wtag')
    s.preprocess()
    h = s.histories[0]
    f = s.generate_feature_vector(h)
    fs = s.generate_expected_count_features(h)

def main():
    s = textProcessor('data/train1.wtag')
    s.preprocess()
    model = MEMM(s)
    model.fit(1)

    return


if __name__ == "__main__":
    main()