from main import fill_file, hyper_parameters_model1, hyper_parameters_model2

fill_file(['data/train1.wtag', 'data/test1.wtag'], 'data/comp1.words', 'data/comp_m1_both_204506349.wtag',
          hyper_parameters_model1, load_weights=True, weight_file='model1_comp_weights')
fill_file(['data/train2.wtag'], 'data/comp2.words', 'data/comp_m2_204506349.wtag', hyper_parameters_model2,
          load_weights=True, weight_file='model2_weights')