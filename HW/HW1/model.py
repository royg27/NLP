import numpy as np
from textProcessor import textProcessor
import settings
from scipy.optimize import fmin_l_bfgs_b


class MEMM:
    def __init__(self, textProcessor : textProcessor, lamda=0.1, lr=0.01, sigma = 0.001):
        self.processor = textProcessor
        self.lamda = lamda
        self.lr = lr
        #   Standard deviation initialization for the weights vector
        self.v = sigma * np.random.randn(self.processor.f_length)

    # def our_softmax(self, f_x_y, f_x_y_tags):
    #     """
    #
    #     :param f_x_y:
    #     :param f_x_y_tags:
    #     :return:
    #     """
    #     nominator = np.exp(np.dot(self.v, f_x_y))
    #     # the loop calcs all the possible exp dot for every tag
    #     denominator = np.sum([np.exp(np.dot(self.v, f_x_y_t)) for f_x_y_t in f_x_y_tags])
    #     soft_max_res = nominator/denominator
    #     assert (soft_max_res >= 0 and soft_max_res <=1)
    #     return soft_max_res


    def vectorized_softmax(self,F,w_i):
        """

        :param F: is N*M where each row is a sparse vector
        :return:
        """
        print("vectorized_softmax")
        print("w_i = ",w_i)
        print("none zeros in F' = ", F.count_nonzero()/F.shape[0])
        F_V = F.dot(w_i)
        exp_F_V = np.exp(F_V,dtype=np.double)
        print("max(exp_F_V)", np.max(exp_F_V), "max(w_i) = ",np.max(w_i))
        print("type(exp_F_V[0]) = ", type(exp_F_V[0]))
        sum_exp_F_V = np.sum(exp_F_V)
        print("sum_exp_F_V = ", sum_exp_F_V)
        softmax_val = exp_F_V/sum_exp_F_V
        print("softmax_val[0] = ",softmax_val[0])
        return softmax_val

    def vectorized_calc_gradient(self, empirical_counts, F, h_tag_len, w_i):
        """
        :param empirical_counts:
        :param F:
        :return:
        """
        print("vectorized_calc_gradient")
        P = self.vectorized_softmax(F, w_i)[:,np.newaxis]
        expected_counts = F.multiply(P)
        expected_counts = expected_counts.sum(axis=0)
        expected_counts = np.ravel(expected_counts)
        grad = empirical_counts - expected_counts - (self.lamda * self.v)
        return grad

    # def calc_gradient(self, f_x_y, f_x_y_tags):
    #     """
    #     :param f_x_y: f(x,y)
    #     :param f_x_y_tags: Vector of f(x,y') where y'!=y, for all possible tags of y
    #     :return: Vector that equals to dL/dv
    #     """
    #     emprical_counts = f_x_y
    #     expected_counts = np.zeros(f_x_y.shape[0])
    #     for f in f_x_y_tags:
    #         expected_counts += (self.our_softmax(f, f_x_y_tags) * f)
    #     grad = emprical_counts - expected_counts - (self.lamda * self.v)
    #     return grad

    def vectorized_calc_likelihood(self, F, F_tag, h_tag_len, w_i):
        print("vectorized_calc_likelihood")
        linear_term = F.multiply(w_i)
        linear_term = linear_term.sum()
        regularization_term = 0.5 * self.lamda * (np.linalg.norm(w_i)**2)
        print("w_i = ",w_i)
        f_v = F_tag.dot(w_i)
        print("f_v = ",f_v)
        e_f_v = np.exp(f_v)
        # reshape as (-1,tags)
        reshaped_e_f_v = e_f_v.reshape((-1,h_tag_len))
        summed_reshaped = reshaped_e_f_v.sum(axis=1)
        log_summed = np.log(summed_reshaped)
        normalization_term = log_summed.sum(axis=0)
        likelihood = linear_term - normalization_term - regularization_term
        print("likelihood = ",likelihood)
        return likelihood

    def calc_objective_per_iter(self, w_i, *args):
        """
            Calculate max entropy likelihood for an iterative optimization method
            :param w_i: weights vector in iteration i
            :param arg_i: arguments passed to this function, such as lambda hyperparameter for regularization

                The function returns the Max Entropy likelihood (objective) and the objective gradient
        """
        print("calc_objective_per_iter")
        empirical_count = args[0]
        F_tag =args[1]
        F = args[2]
        h_tag_len = args[3]
        grad = self.vectorized_calc_gradient(empirical_count, F_tag, h_tag_len, w_i)
        print("|grad| = ", np.linalg.norm(grad))
        likelihood = self.vectorized_calc_likelihood(F, F_tag, h_tag_len, w_i)
        return (-1) * likelihood, (-1) * grad



    def fit2(self):
        H = self.processor.histories
        H_tag = self.processor.generate_H_tag()
        F = self.processor.generate_F(H)
        F_tag = self.processor.generate_F(H_tag)
        #

        print("F.shape ",F.shape)
        print("F_tag.shape ", F_tag.shape)
        empirical_counts = F.sum(axis=0)
        print(empirical_counts.shape)
        print(empirical_counts)

        #   set params for Gradient Ascent library function
        args = (empirical_counts, F_tag, F, len(self.processor.tags_set))
        w_0 = np.zeros(self.processor.f_length, dtype=np.float64)
        optimal_params = fmin_l_bfgs_b(func=self.calc_objective_per_iter, x0=w_0, args=args, maxiter=1000, iprint=99)
        weights = optimal_params[0]

    # def fit(self, num_epochs):
    #     for epoch in range(num_epochs):
    #         grad = np.zeros(self.processor.f_length)
    #         # run over sentences
    #         for idx,H in enumerate(self.processor.histories[0:1]):
    #             # run over histories of a given sentence
    #             print("working on sentence ",idx)
    #             for i, h in enumerate(H):
    #                 print("working on history ", i)
    #                 f = self.processor.generate_feature_vector(h)
    #                 F = self.processor.generate_expected_count_features(h)
    #                 if settings.use_vectorized_sparse:
    #                     cur_grad = self.vectorized_calc_gradient(f, F)
    #                     assert np.count_nonzero(cur_grad) != 0
    #                     grad += cur_grad
    #                 else:
    #                     grad += self.calc_gradient(f,F)
    #
    #             #   update v vector after a batch - a sentence
    #             delta_v = (self.lr * grad)
    #             assert np.count_nonzero(delta_v) != 0
    #             self.v += (self.lr * grad)
    #     # load weights to csv files
    #     savetxt('weights.csv',self.v, delimiter=',')
