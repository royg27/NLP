import numpy as np
from textProcessor import textProcessor
from scipy.optimize import fmin_l_bfgs_b
from numpy import savetxt
from numpy import loadtxt


class MEMM:
    def __init__(self, textProcessor : textProcessor, lamda=0.01, sigma=0.001):
        #np.random.seed(13)
        self.processor = textProcessor
        self.lamda = lamda
        #   Standard deviation initialization for the weights vector
        self.v = sigma * np.random.randn(self.processor.f_length)

    def calc_objective_per_iter(self, w_i, *args):
        #   extract the parameters from args
        empirical_count = args[0]
        F_tag = args[1]
        F = args[2]
        h_tag_len = args[3]
        #   calculate gradient
        grad = self.vectorized_calc_gradient(empirical_count, F_tag, h_tag_len, w_i)

        #   calculate likelihood
        likelihood = self.vectorized_calc_likelihood(F, F_tag, h_tag_len, w_i)
        return (-1) * likelihood, (-1) * grad

    def vectorized_softmax(self,F, w_i, h_tag_len):
        """
        calculates the probability of every f in F'
        the probability is based on running over all possible y' per history
        we are doing it without loops while exploiting the fact that each history h has fixed number of h'
        :param F: feature vectors per every h'
        :param w_i: current weights
        :param h_tag_len: amount of h' per h
        :return: probability for every f'
        """
        F_V = F.dot(w_i)
        e_f_v = np.exp(F_V)
        # reshape in such a way that every row contains values per one history in train set
        reshaped_e_f_v = e_f_v.reshape((-1, h_tag_len))
        row_sum = reshaped_e_f_v.sum(axis=1)[:,np.newaxis]
        soft_max_matrix = reshaped_e_f_v / row_sum
        # now, each cell corresponds to one f' and its value is its probability
        soft_max = np.ravel(soft_max_matrix)
        return soft_max

    def vectorized_calc_likelihood(self, F, F_tag, h_tag_len, w_i):
        """
        will calculate the Likelihood function that we would like to maximize
        :param F: sparse matrix, each row is a feature vector of history from the data set
        :param F_tag: sparse matrix, each row is a feature vector of history(x,y')
        :param h_tag_len: amount of h' per h
        :param w_i: current weights vector
        :return: the likelihood value
        """
        linear_term = F.multiply(w_i)
        linear_term = linear_term.sum()
        regularization_term = 0.5 * self.lamda * (np.linalg.norm(w_i)**2)
        f_v = F_tag.dot(w_i)
        e_f_v = np.exp(f_v)
        # reshape as (-1,tags)
        reshaped_e_f_v = e_f_v.reshape((-1,h_tag_len))
        summed_reshaped = reshaped_e_f_v.sum(axis=1)
        log_summed = np.log(summed_reshaped)
        normalization_term = log_summed.sum(axis=0)
        likelihood = linear_term - normalization_term - regularization_term
        return likelihood

    def vectorized_calc_gradient(self, empirical_counts, F, h_tag_len, w_i):
        """
        :param empirical_counts: a row vector that represents empirical counts
        :param F: f for every possible h' where h' is a variant of h that is seen in data set
        :return: gradient vector
        """
        P = self.vectorized_softmax(F, w_i, h_tag_len)[:,np.newaxis]
        expected_counts = F.multiply(P)

        expected_counts = expected_counts.sum(axis=0)
        expected_counts = np.ravel(expected_counts)
        grad = empirical_counts - expected_counts - (self.lamda * self.v)
        return grad

    def fit(self,load_weights=True):
        if load_weights:
            self.v = loadtxt('weights.csv')
            return
        H = self.processor.histories
        H_tag = self.processor.generate_H_tag()
        F = self.processor.generate_F(H)
        F_tag = self.processor.generate_F(H_tag)
        empirical_counts = F.sum(axis=0)
        args = (empirical_counts, F_tag, F, len(self.processor.tags_set))
        w_0 = np.zeros(self.processor.f_length, dtype=np.float64)
        optimal_params=fmin_l_bfgs_b(func=self.calc_objective_per_iter, x0=w_0, args=args, maxiter=1000, iprint=10)
        self.v = optimal_params[0]
        savetxt('weights.csv', self.v, delimiter=',')
        print("v = ",self.v)

    # def predict(self,file_path):
        # s = textProcessor(file_path)
        # s.preprocess()
        # H = s.histories
        # H_tag = s.generate_H_tag()
        # F = s.generate_feature_vector()
        # F_tag = s.generate_feature_vector(H_tag)
        # for sentence in sentences:
        #     viterbi(sentence)


    def viterbi_roy(self, sentence=['The','Treasury','is', 'still','working','out'], beam=3):
        sample = 5
        sentence = self.processor.sentences[sample]
        pi = np.zeros((len(sentence) + 1, len(self.processor.tags_set), len(self.processor.tags_set)))
        bp = np.zeros((len(sentence) + 1, len(self.processor.tags_set), len(self.processor.tags_set)))
        #   init pi(0),bp(0
        star_idx = np.where(self.processor.tags_set == '*')[0][0]
        pi[0, star_idx, star_idx] = 1
        bp[0,star_idx,star_idx] = star_idx
        relevant_idx = []
        relevant_tags = []
        tags_list = self.processor.tags_set
        relevant_idx_u = []
        relevant_tags_u = []
        relevant_idx_t = []
        relevant_tags_t = []
        for idx, word in enumerate(sentence):
            if idx==0:
                relevant_idx_u = [star_idx]
                relevant_tags_u = ['*']
                relevant_idx_t = [star_idx]
                relevant_tags_t = ['*']
            else:
                relevant_idx_t = np.arange(len(self.processor.tags_set))
                relevant_tags_t = tags_list
            for u_idx, u in zip(relevant_idx_u, relevant_tags_u):
                v = -np.inf * np.ones(len(tags_list))
                t_bp = np.zeros(len(tags_list))
                for t_idx, t in zip(relevant_idx_t, relevant_tags_t):
                    h_tag = self.processor.generate_h_tag_for_word_roy(word,t,u)
                    F_tag = self.processor.generate_F(h_tag)
                    q = self.vectorized_softmax(F_tag, self.v, len(tags_list))
                    #   q is for all possible v for given t,u
                    val = q * pi[idx,t_idx,u_idx]
                    indexes_to_update = v<val
                    v[indexes_to_update] = val[indexes_to_update]
                    t_bp[indexes_to_update] = t_idx
                pi[idx+1,u_idx,:] = v
                bp[idx+1,u_idx,:] = t_bp
            #   beam search
            pi_col_max = np.max(pi[idx + 1, :, :],axis=0)
            relevant_idx_u = np.argsort(pi_col_max)[-beam:]
            relevant_tags_u = self.processor.tags_set[relevant_idx_u]
            #
        # backpass
        N, rows,cols = pi.shape
        max_t_n_1, max_t_n = 0,0
        max = 0
        for row in range(rows):
            for col in range(cols):
                if pi[-1,row,col]>max:
                    max = pi[-1,row,col]
                    max_t_n_1, max_t_n = row, col
        tags = np.zeros(N)
        tags[-2:] = [max_t_n_1, max_t_n]
        indexes = np.arange(3,N+1)    #   (3,4,....,N)
        for j,true_i in enumerate(indexes):
            idx = N - true_i    #   (N-3,N-4,..1)
            tags[idx] = bp[idx+2,int(tags[idx+1]),int(tags[idx+2])]
        ret_val = []
        for t in tags:
            ret_val.append(tags_list[int(t)])
        print(sentence)
        print("Ground truth:")
        print(self.processor.tags[sample])
        print("predicted:")
        print(ret_val[1:])
        pred = np.array(ret_val[1:])
        grount_truth = np.array(self.processor.tags[sample])
        print("acc = ", (pred == grount_truth).sum()/grount_truth.shape[0])





    def viterbi(self, sentence=['The','Treasury','is', 'still'], beam=3):
        pi = np.zeros((len(sentence)+1, len(self.processor.tags_set), len(self.processor.tags_set)))
        #   init pi(0)
        star_idx = np.where(self.processor.tags_set == '*')[0][0]
        pi[0,star_idx,star_idx] = 1
        relevant_idx = []
        relevant_tags = []
        for idx, word in enumerate(sentence):
            #   create all possible histories
            if idx==0:
                relevant_idx = [star_idx]
                relevant_tags = ['*']
            for row,u in zip(relevant_idx, relevant_tags):    # when implementing beam, run only on top k
                # u is the t_1, last tag
                for col,v in enumerate(self.processor.tags_set):
                    # v is the current tag
                    h_tag = self.processor.generate_h_tag_for_word(word,u,v)
                    F_tag = self.processor.generate_F(h_tag)
                    q = self.vectorized_softmax(F_tag, self.v, len(self.processor.tags_set))
                    pi_v = pi[idx, :, row]
                    val = np.multiply(q, pi_v)
                    pi[idx + 1, row, col] = np.max(val)
            #   beam search
            relevant_idx = np.argsort(pi[idx + 1, row, :])[-beam:]
            relevant_tags = self.processor.tags_set[relevant_idx]
            print(pi[idx+1,:,:])

                    # max_val = -1 * np.inf
                    # for t_idx, tag in enumerate(self.processor.tags_set):
                    #     val = pi[idx, t_idx, u] * q[t_idx]
                    #     if val > max_val:
                    #         max_val = val
                    # pi[idx+1,row,col] = max_val
        return



    # def fit2(self):
    #     H = self.processor.histories
    #     H_tag = self.processor.generate_H_tag()
    #     F = self.processor.generate_F(H)
    #     F_tag = self.processor.generate_F(H_tag)
    #     #
    #
    #     print("F.shape ",F.shape)
    #     print("F_tag.shape ", F_tag.shape)
    #     empirical_counts = F.sum(axis=0)
    #     print(empirical_counts.shape)
    #     print(empirical_counts)
    #
    #     #   set params for Gradient Ascent library function
    #     args = (empirical_counts, F_tag, F, len(self.processor.tags_set))
    #     w_0 = np.zeros(self.processor.f_length, dtype=np.float64)
    #     optimal_params = fmin_l_bfgs_b(func=self.calc_objective_per_iter, x0=w_0, args=args, maxiter=1000, iprint=99)
    #     weights = optimal_params[0]

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


    #
    # def calc_objective_per_iter(self, w_i, *args):
    #     """
    #         Calculate max entropy likelihood for an iterative optimization method
    #         :param w_i: weights vector in iteration i
    #         :param arg_i: arguments passed to this function, such as lambda hyperparameter for regularization
    #
    #             The function returns the Max Entropy likelihood (objective) and the objective gradient
    #     """
    #     print("calc_objective_per_iter")
    #     empirical_count = args[0]
    #     F_tag =args[1]
    #     F = args[2]
    #     h_tag_len = args[3]
    #     grad = self.vectorized_calc_gradient(empirical_count, F_tag, h_tag_len, w_i)
    #     print("|grad| = ", np.linalg.norm(grad))
    #     likelihood = self.vectorized_calc_likelihood(F, F_tag, h_tag_len, w_i)
    #     return (-1) * likelihood, (-1) * grad

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


    # def vectorized_softmax(self,F,w_i):
    #     """
    #
    #     :param F: is N*M where each row is a sparse vector
    #     :return:
    #     """
    #     print("vectorized_softmax")
    #     print("w_i = ",w_i)
    #     print("none zeros in F' = ", F.count_nonzero()/F.shape[0])
    #     F_V = F.dot(w_i)
    #     exp_F_V = np.exp(F_V,dtype=np.double)
    #     print("max(exp_F_V)", np.max(exp_F_V), "max(w_i) = ",np.max(w_i))
    #     print("type(exp_F_V[0]) = ", type(exp_F_V[0]))
    #     sum_exp_F_V = np.sum(exp_F_V)
    #     print("sum_exp_F_V = ", sum_exp_F_V)
    #     softmax_val = exp_F_V/sum_exp_F_V
    #     print("softmax_val[0] = ",softmax_val[0])
    #     return softmax_val
    #
    # def vectorized_calc_gradient(self, empirical_counts, F, h_tag_len, w_i):
    #     """
    #     :param empirical_counts:
    #     :param F:
    #     :return:
    #     """
    #     print("vectorized_calc_gradient")
    #     P = self.vectorized_softmax(F, w_i)[:,np.newaxis]
    #     expected_counts = F.multiply(P)
    #     expected_counts = expected_counts.sum(axis=0)
    #     expected_counts = np.ravel(expected_counts)
    #     grad = empirical_counts - expected_counts - (self.lamda * self.v)
    #     return grad

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


