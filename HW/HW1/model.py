import numpy as np
from textProcessor import textProcessor

class MEMM:
    def __init__(self, textProcessor : textProcessor, lamda = 0.01, lr = 0.01, sigma = 0.001):
        self.processor = textProcessor
        self.lamda = lamda
        self.lr = lr
        #   Standard deviation initialization for the weights vector
        self.v = sigma * np.random.randn(self.processor.f_length)

    def our_softmax(self, f_x_y, f_x_y_tags):
        """

        :param f_x_y:
        :param f_x_y_tags:
        :return:
        """
        nominator = np.exp(np.dot(self.v, f_x_y))
        # the loop calcs all the possible exp dot for every tag
        denominator = np.sum([np.exp(np.dot(self.v, f_x_y_t)) for f_x_y_t in f_x_y_tags])
        soft_max_res = nominator/denominator
        assert (soft_max_res >= 0 and soft_max_res <=1)
        return soft_max_res

    def calc_gradient(self, f_x_y, f_x_y_tags):
        """
        :param f_x_y: f(x,y)
        :param f_x_y_tags: Vector of f(x,y') where y'!=y, for all possible tags of y
        :return: Vector that equals to dL/dv
        """
        emprical_counts = f_x_y
        expected_counts = np.zeros(f_x_y.shape[0])
        for f in f_x_y_tags:
            expected_counts += (self.our_softmax(f, f_x_y_tags) * f)
        grad = emprical_counts - expected_counts - (self.lamda * self.v)
        return grad

    def fit(self, num_epochs):
        for epoch in range(num_epochs):
            grad = np.zeros(self.processor.f_length)
            # run over sentences
            for H in self.processor.histories:
                # run over histories of a given sentence
                for h in H:
                    f_x_y = self.processor.generate_feature_vector(h)
                    f_x_y_tags = self.processor.generate_expected_count_features(h)
                    grad += self.calc_gradient(f_x_y, f_x_y_tags)
                #   update v vector after a batch - a sentence
                self.v += (self.lr * grad)
