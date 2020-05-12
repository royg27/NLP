import numpy as np
from textProcessor import textProcessor
from scipy.optimize import fmin_l_bfgs_b
import re
from sklearn.metrics import confusion_matrix
import pickle
import pandas as pd


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
        # trick for numerical stability
        # reshape in such a way that every row contains values per one history in train set
        F_V_reshaped = F_V.reshape((-1, h_tag_len))
        F_V_max = F_V_reshaped.max(axis=1)
        F_V_reshaped -= F_V_max[:,np.newaxis]
        reshaped_e_f_v = np.exp(F_V_reshaped)
        # reshaped_e_f_v = e_f_v.reshape((-1, h_tag_len))
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

    def fit(self, load_weights=True, weights_path="trained_weights"):
        if load_weights:
            with open(weights_path, 'rb') as f:
                optimal_params = pickle.load(f)
            self.v = optimal_params[0]
            return
        H = self.processor.histories
        H_tag = self.processor.generate_H_tag()
        F = self.processor.generate_F(H)
        F_tag = self.processor.generate_F(H_tag)
        empirical_counts = F.sum(axis=0)
        args = (empirical_counts, F_tag, F, len(self.processor.tags_set))
        w_0 = np.zeros(self.processor.f_length, dtype=np.float64)
        optimal_params=fmin_l_bfgs_b(func=self.calc_objective_per_iter, x0=w_0, args=args, maxiter=1000, iprint=0)
        with open(weights_path, 'wb') as f:
            pickle.dump(optimal_params, f)
        self.v = optimal_params[0]

    def accuracy(self, Y, Y_pred):
        """
        :param Y: list of ground truth tags
        :param Y_pred: list of our model's tags
        :return: accuracy score per word
        """
        correct_count = 0
        total_count = 0
        for idx_sentence, sentence_tags in enumerate(Y):
            for idx_word, word_tag in enumerate(sentence_tags):
                if word_tag == Y_pred[idx_sentence][idx_word]:
                    correct_count += 1
                total_count += 1
        return correct_count/total_count

    def generate_predicted_file(self, file, output_file, beam=3):
        f = open(file, 'r')
        f_output = open(output_file, 'w')
        for line in f.readlines():
            sentence = re.split(' |\n', line)
            del sentence[-1]
            tags = self.viterbi(sentence, beam=beam)
            tagged_sentence = []
            for idx, (word, tag) in enumerate(zip(sentence, tags)):
                if idx == 0:
                    tagged_sentence.append(word + "_" + tag)
                else:
                    tagged_sentence.append(" " + word + "_" + tag)
            tagged_sentence.append("\n")
            f_output.writelines(tagged_sentence)
        f.close()
        f_output.close()


    def predict(self, file_path, beam=3, num_sentences=-1):
        s = textProcessor([file_path])
        s.preprocess()
        if num_sentences == -1:
            num_sentences = len(s.sentences)
        av_acc = 0
        y_pred = []
        for idx, sentence in enumerate(s.sentences):
            if idx >= num_sentences:
                break
            y_pred.append(self.viterbi(sentence, beam=beam))
        return y_pred

    def viterbi(self, sentence=['In', 'other', 'words', ',', 'it', 'was'], beam=3):
        pi = np.zeros((len(sentence) + 1, len(self.processor.tags_set), len(self.processor.tags_set)))
        bp = np.zeros((len(sentence) + 1, len(self.processor.tags_set), len(self.processor.tags_set)))
        #   init pi(0),bp(0)
        star_idx = np.where(self.processor.tags_set == '*')[0][0]
        pi[0, star_idx, star_idx] = 1
        bp[0,star_idx,star_idx] = star_idx
        relevant_idx = []
        relevant_tags = []
        tags_list = self.processor.tags_set
        relevant_idx_u = []
        relevant_tags_u = []
        for idx, word in enumerate(sentence):
            prev_word = sentence[idx-1] if idx>0 else '*'
            next_word = sentence[idx+1] if idx+1<len(sentence) else '*'
            if idx==0:
                relevant_idx_u = [star_idx]
                relevant_tags_u = ['*']
                relevant_idx_t = [star_idx]
                relevant_tags_t = ['*']
            # else:
            #     relevant_idx_t = np.arange(len(self.processor.tags_set))
            #     relevant_tags_t = tags_list
            for u_idx, u in zip(relevant_idx_u, relevant_tags_u):
                v = -np.inf * np.ones(len(tags_list))
                t_bp = np.zeros(len(tags_list))
                for t_idx, t in zip(relevant_idx_t, relevant_tags_t):
                    h_tag = self.processor.generate_h_tag_for_word_roy(word, t, u, prev_word, next_word)
                    F_tag = self.processor.generate_F(h_tag)
                    q = self.vectorized_softmax(F_tag, self.v, len(tags_list))
                    #   q is for all possible v for given t,u
                    val = q * pi[idx,t_idx,u_idx]
                    indexes_to_update = v<val
                    #   update v according to its maximal value, if the current t increased on of its values
                    v[indexes_to_update] = val[indexes_to_update]
                    #   update the the t caused the updates
                    t_bp[indexes_to_update] = t_idx
                pi[idx+1,u_idx,:] = v
                bp[idx+1,u_idx,:] = t_bp
            #   beam search
            pi_col_max = np.max(pi[idx + 1, :, :],axis=0)
            #   save prev relevant u as t
            relevant_idx_t = relevant_idx_u
            relevant_tags_t = relevant_tags_u
            #
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
        return ret_val[1:]

    def confusion_matrix_roy(self, Y, Y_pred):
        tags_to_show = self.get_worst_tags_roy(Y, Y_pred)
        labels = self.processor.tags_set
        y_test = [y for x in Y for y in x]
        y_pred = [y for x in Y_pred for y in x]
        conf_mat = confusion_matrix(y_test, y_pred, labels)
        truncated_conf_mat = conf_mat[tags_to_show, :]
        df = pd.DataFrame(truncated_conf_mat, columns=labels, index=labels[tags_to_show])
        df.to_html('conf_mat.html')
        print(conf_mat)
        ##  make nice graph
        # disp = ConfusionMatrixDisplay(conf_mat, labels)
        # disp = disp.plot(include_values=False ,cmap='viridis', ax=None, xticks_rotation='horizontal')
        # plt.show()

    def get_worst_tags_roy(self, Y, Y_pred, num_worst = 10):
        """
        :param Y: GT of the tags
        :param Y_pred: The predicted tags
        :return: A list of the worst predicted tags
        """
        possible_tags = self.processor.tags_set
        word_acc = dict() # np.zeros(shape=len(possible_tags))
        for tag in possible_tags:
            word_acc[tag] = (0, 0)

        for idx_sentence, sentence_tags in enumerate(Y):
            for idx_word, word_tag in enumerate(sentence_tags):
                curr_count, curr_right = word_acc[word_tag]
                if word_tag == Y_pred[idx_sentence][idx_word]:
                    curr_right += 1
                curr_count += 1
                word_acc[word_tag] = (curr_count, curr_right)

        tag_res = np.zeros(shape=len(possible_tags))
        for idx, tag in enumerate(possible_tags):
            count, right = word_acc[tag]
            if count == 0:
                continue
            tag_res[idx] = 1 - (right/count)

        worse_tags_idx = np.argsort(tag_res)[-num_worst:]
        print(possible_tags[worse_tags_idx])
        return worse_tags_idx