import numpy as np
from textProcessor import textProcessor
import settings
from numpy import savetxt

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


    def vectorized_softmax(self,F):
        """

        :param F: is N*M where each row is a sparse vector
        :return:
        """
        #assert self.v.shape[0] == F.shape[1]
        shape = F.shape
        F_V = F.dot(self.v)
        #assert F_V.shape[0] == F.shape[0]
        exp_F_V = np.exp(F_V)
        sum_exp_F_V = np.sum(exp_F_V)
        softmax_val = exp_F_V/sum_exp_F_V
        return softmax_val

    def vectorized_calc_gradient(self, f, F):
        """
        :param f:
        :param F:
        :return:
        """
        emprical_counts = f
        P = self.vectorized_softmax(F)[:,np.newaxis]
        expected_counts = F.multiply(P)
        expected_counts = expected_counts.sum(axis=0)
        expected_counts = np.ravel(expected_counts)
        grad = emprical_counts - expected_counts - (self.lamda * self.v)
        return grad

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
            for idx,H in enumerate(self.processor.histories[0:1]):
                # run over histories of a given sentence
                print("working on sentence ",idx)
                for i, h in enumerate(H):
                    print("working on history ", i)
                    f = self.processor.generate_feature_vector(h)
                    F = self.processor.generate_expected_count_features(h)
                    if settings.use_vectorized_sparse:
                        cur_grad = self.vectorized_calc_gradient(f, F)
                        assert np.count_nonzero(cur_grad) != 0
                        grad += cur_grad
                    else:
                        grad += self.calc_gradient(f,F)

                #   update v vector after a batch - a sentence
                delta_v = (self.lr * grad)
                assert np.count_nonzero(delta_v) != 0
                self.v += (self.lr * grad)
        # load weights to csv files
        savetxt('weights.csv',self.v, delimiter=',')
