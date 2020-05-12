from textProcessor import textProcessor
from model import MEMM
import time
import numpy as np
import unit_tests
hyper_parameters_model1 = {"thr": 3, "thr_2": 7, "lamda": 2}
hyper_parameters_model2 = {"thr": 3, "thr_2": 5, "lamda": 0.5}


# TODO change so will get weight_file - should have weights for combined train+test for model 1
def fill_file(train_file_list, file_to_predict, tagged_file_name,hyper_parameters, weight_file, load_weights=False):
    s = textProcessor(train_file_list, thr=hyper_parameters["thr"], thr_2=hyper_parameters["thr_2"])
    s.preprocess()
    model = MEMM(s, lamda=hyper_parameters["lamda"], sigma=0.001)
    model.fit(load_weights, weights_path=weight_file)
    predict_file = file_to_predict
    model.generate_predicted_file(predict_file, tagged_file_name, beam=3)


def hyperparameter_tuning():
    num_sentences = 100
    processor_thr = [(3,7), (3,5), (5,7)]
    model_lambda = [2, 5]
    max_val = 0
    best_lmbda = -1
    best_thr = -1
    s_val = textProcessor(['data/test1.wtag'])
    s_val.preprocess()
    Y = s_val.tags[0:num_sentences]
    for thr in processor_thr:
        for lmbda in model_lambda:
            print("testing thr =",thr, " lambda = ", lmbda)
            s = textProcessor(['data/train1.wtag'],thr=thr[0], thr_2=thr[1])
            s.preprocess()
            model = MEMM(s,lamda=lmbda)
            t = time.time()
            model.fit(False)
            end = time.time()
            print("fit time = ", (end - t))
            t = time.time()
            y_pred = model.predict('data/test1.wtag', num_sentences=num_sentences, beam=3)
            end = time.time()
            print("prediction time = ", (end-t))
            cur_acc = model.accuracy(Y, y_pred)
            print("cur_acc = ", cur_acc)
            if cur_acc > max_val:
                best_lmbda = lmbda
                best_thr = thr
                max_val = cur_acc
    print("best acc = ",max_val, " thr = ", best_thr, " lambda = ", best_lmbda, " use extra = ")
    hyper_parameters_model1["thr"] = best_thr[0]
    hyper_parameters_model1["thr_2"] = best_thr[1]
    hyper_parameters_model1["lamda"] = best_lmbda


def training_part(hyper_params, train_file_list, weight_file, load_weights=False):
    print("training...")
    processor = textProcessor(train_file_list, thr=hyper_params["thr"], thr_2=hyper_params["thr_2"])
    processor.preprocess()
    print("|f100| = ", len(processor.feature_100))
    print("|f101| = ", len(processor.feature_101))
    print("|f102| = ", len(processor.feature_102))
    print("|f103| = ", len(processor.feature_103))
    print("|f104| = ", len(processor.feature_104))
    print("|f105| = ", len(processor.feature_105))
    print("|f106| = ", len(processor.feature_106))
    print("|f_is_numeric| = ", len(processor.feature_contains_numbers))
    print("|f_start_capital| = ", len(processor.feature_start_cap))
    print("|f_all_capital| = ", len(processor.feature_all_caps))
    print("|f_contains_hyphen| = ", len(processor.feature_contains_hyphen))
    model = MEMM(processor,lamda=hyper_params["lamda"])
    start = time.time()
    model.fit(load_weights,weights_path=weight_file)
    end = time.time()
    print("training time = ", end - start)
    return model


def generate_confusion_matrix(Y=None,Y_pred=None):
    if Y is None or Y_pred is None:
        s = textProcessor(['data/train1.wtag'], thr=hyper_parameters_model1["thr"], thr_2=hyper_parameters_model1["thr_2"])
        s.preprocess()
        num_sentences_to_tag = 1000
        model = MEMM(s, lamda=hyper_parameters_model1["lamda"])
        model.fit(load_weights=True)
        start = time.time()
        Y_pred = model.predict('data/test1.wtag', num_sentences=num_sentences_to_tag, beam=3)
        end = time.time()
        print("predict time : ",end-start)
        s_val = textProcessor(['data/test1.wtag'], thr=3)
        s_val.preprocess()
        Y = s_val.tags[:num_sentences_to_tag]
        model.confusion_matrix_roy(Y, Y_pred)
    else:
        s = textProcessor(['data/train1.wtag'], thr=hyper_parameters_model1["thr"], thr_2=hyper_parameters_model1["thr_2"])
        s.preprocess()
        model = MEMM(s, lamda=hyper_parameters_model1["lamda"])
        model.fit(load_weights=True)
        model.confusion_matrix_roy(Y, Y_pred)



def k_folds_cross_validation_train2(thr=3,thr_2=7,lamda=2):
    files = ['data/train2_1.wtag','data/train2_2.wtag','data/train2_3.wtag','data/train2_4.wtag','data/train2_5.wtag']
    cur_acc = 0
    for i, test_file in enumerate(files):
        s_test = textProcessor([test_file], thr=thr, thr_2=thr_2)
        s_test.preprocess()
        Y = s_test.tags
        train_files = []
        for j, train_file in enumerate(files):
            if j!=i:
                train_files.append(train_file)
        #   train on train_files
        s_train = textProcessor(train_files, thr=thr, thr_2=thr_2)
        s_train.preprocess()
        model = MEMM(s_train, lamda=lamda)
        model.fit(False)
        # predict
        y_pred = model.predict(test_file, num_sentences=50, beam=3)
        cur_acc += model.accuracy(Y, y_pred)
    return cur_acc/5.


def inference(model):
    s_val = textProcessor(['data/test1.wtag'])
    s_val.preprocess()
    Y = s_val.tags
    start = time.time()
    y_pred = model.predict('data/test1.wtag', num_sentences=1000, beam=3)
    end = time.time()
    print("inference time : ", end- start)
    cur_acc = model.accuracy(Y, y_pred)
    print("model's accuracy: ", cur_acc)
    return Y, y_pred

def hyper_parameter_tuning_model2():
    thr = [(3, 5), (3, 7), (5, 5), (5, 7)]
    lamdas = [0.5,1,2,5]
    best_thr = -1
    best_thr_2 = -1
    best_lamda = -1
    best_acc = -1
    for thr, thr_2 in thr:
        for lamda in lamdas:
            print("checking thr = ",thr, " thr_2 = ", thr_2, " lamda = ", lamda)
            acc = k_folds_cross_validation_train2(thr,thr_2,lamda)
            print("acc = ", acc)
            if acc>best_acc:
                best_thr = thr
                best_thr_2 = thr_2
                best_lamda = lamda
                best_acc = acc
    print("best acc = ", best_acc, "params = ", best_thr, " ", best_thr_2, " ", best_lamda)
    hyper_parameters_model2["thr"] = best_thr
    hyper_parameters_model2["thr_2"] = best_thr_2
    hyper_parameters_model2["lamda"] = best_lamda



def main():
    # hyper-parameters_tunning
    # hyperparameter_tuning()
    # hyper_parameter_tuning_model2()
    # model 1:
    print("model 1: ")
    trained_model1 = training_part(hyper_parameters_model1, ['data/train1.wtag'], weight_file='model1_weights', load_weights=False)
    Y, y_pred = inference(trained_model1)
    generate_confusion_matrix(Y, y_pred)
    # TODO think if using both texts to generate comp1 tags makes sense
    fill_file(['data/train1.wtag'], 'data/comp1.words', 'data/comp_m1_204506349.wtag',
              hyper_parameters_model1, load_weights=True, weight_file='model1_weights')

    print("model 2: ")
    print("k-folds cross validation on model2 : ")
    print("estimated accuracy : ", k_folds_cross_validation_train2(**hyper_parameters_model2))
    trained_model2 = training_part(hyper_parameters_model2, ['data/train2.wtag'], weight_file='model2_weights', load_weights=False)
    fill_file(['data/train2.wtag'], 'data/comp2.words', 'data/comp_m2_204506349.wtag', hyper_parameters_model2,
              load_weights=True, weight_file='model2_weights')
    return


if __name__ == "__main__":
    main()