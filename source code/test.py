import pandas as pd
import numpy as np
from recommender.core import Recommender


""" These are the codes that I used to test the Recommender class 
    and to generate keyword recommendation for the test file
"""

""" These functions are for testing purpose """


# read sample data to dataframe from the given file_path
# Input: sample data file directory; Output: log data in dataframes by category
def read_data(file_path):
    data = pd.read_csv(file_path, sep=',')
    data.columns = ['product', 'category', 'query', 'event', 'date']
    categories = data['category'].unique()
    data_cat = {}
    for cat in categories:
        data_cat[cat] = data.loc[data['category'] == cat]
    return data_cat


# test the IR based recommendation based on avg top_k recall of query keyword of the click events
# Input: log data of a category, category, top_k value, weight of click rate, number of folds for cross validation
# Output: avg recall, weighted avg recall
def test_ir_recommendation_cv(data, cat, top_k, alpha, k_fold):
    # click_rate_index = build_click_rate_index(data)
    data_imp = data.loc[data['event'] == 'Impression']
    data_clk = data.loc[data['event'] == 'Click']
    # data_clk = data_clk.sample(frac=1).reset_index(drop=True)  # random shuffle the click examples
    fold_len = int(len(data_clk) / k_fold)
    i = 0
    data_clk_folds = {}
    while i < k_fold - 1:
        up = i*fold_len
        down = (i+1)*fold_len
        data_clk_folds[i] = data_clk[up:down]
        i += 1
    data_clk_folds[i] = data_clk[i*fold_len:]

    # test for every fold (indexes built for every round of test, using the rest of data)
    r = 0
    score_sum_rc = 0.0
    score_sum_wrc = 0.0
    while r < k_fold:
        rmd = Recommender()
        rmd.load_data_from_df(data_imp)
        for j in range(k_fold):
            # print(j,' ')
            if j != r:
                rmd.load_data_from_df(data_clk_folds[j])
        # print("indexes built\n")
        fold_score_rc = 0.0
        fold_score_wrc = 0.0
        for index, row in data_clk_folds[r].iterrows():
            result = rmd.recommend_ir_method(row['product'], row['category'], top_k, alpha)

            # data_clk_folds[r][index]['recommendation'] = result
            # evalute if the recommended keywords contains the query
            keyword_list = Recommender.strip(row['query']).split(' ')
            intersection = list(set(result) & set(keyword_list))
            if len(intersection) > 0:
                # cr = get_click_rate(click_rate_index, intersection)
                cr = rmd.get_click_rate(intersection, cat)
                fold_score_rc += len(intersection)/len(keyword_list)
                fold_score_wrc += sum(cr.values())/len(keyword_list)
        # print(data_clk_folds[r].shape)
        fold_score_rc /= len(data_clk_folds[r].index)
        fold_score_wrc /= len(data_clk_folds[r].index)
        score_sum_rc += fold_score_rc
        score_sum_wrc += fold_score_wrc
        r += 1
        # print("fold ", r, " : ", fold_score)

    score_avg_rc = score_sum_rc/k_fold
    score_avg_wrc = score_sum_wrc/k_fold
    return score_avg_rc, score_avg_wrc


# test the IR based recommendation using grid search to find best parameters
# Input: sample file direcotry; Output: None
def test_ir_recommendation_grid_cv(file_path):
    k_list = [5, 10, 15, 20]  # number of k best keywords returned
    alpha_list = [0.0,  0.2, 0.4, 0.6, 0.8, 1.0]  # weight of the click rate score
    fold = 10  # number of folds for the cross_validation
    data = read_data(file_path)
    # testing for each individual category
    for cat in data.keys():
        print(cat, '\n')
        data_cat = data[cat]
        data_cat = data_cat.sample(frac=1).reset_index(drop=True)  # random shuffle the samples
        for k in k_list:
            for alpha in alpha_list:
                score_avg_recall, score_weighted_avg_recall = test_ir_recommendation_cv(data_cat, cat, k, alpha, fold)
                print(k, ' ', alpha, ' ', score_avg_recall, ' ', score_weighted_avg_recall)


# test the ML based recommendation based on avg precision, avg recall, and avg f-score
# Input: log data of a category, regularization strength parameter, number of folds for cross validation
# Output: avg precision, avg recall, avg f-score
def test_ml_recommendation_cv(data, para_c, k_fold):
    fold_len = int(len(data) / k_fold)
    i = 0
    data_folds = {}
    while i < k_fold - 1:
        up = i * fold_len
        down = (i + 1) * fold_len
        data_folds[i] = data[up:down]
        i += 1
    data_folds[i] = data[i * fold_len:]

    # test for every fold (indexes built for every round of test, using the rest of data)
    r = 0  # the test fold
    score_sum_pre = 0.0
    score_sum_rcl = 0.0
    score_sum_f = 0.0
    while r < k_fold:
        rmd = Recommender()
        train_data = pd.DataFrame()
        for j in range(k_fold):
            if j != r:
                train_data = train_data.append(data_folds[j])
                rmd.load_data_from_df(data_folds[j])
        fold_tp = 0.0  # true positive
        fold_pp = 0.0  # predicted positive
        fold_rp = 0.0  # real positive
        model = rmd.recommend_ml_train(train_data, para_c)
        test_x = np.empty([0, 4])  # features
        test_y = []  # class labels
        for index, row in data_folds[r].iterrows():
            x = rmd.get_feature(row['category'], row['product'], row['query'])
            test_x = np.append(test_x, [x], axis=0)
            if row['event'] == 'Click':
                test_y.append(1)
            else:
                test_y.append(0)
        predict_y = model.predict(test_x)
        for index, p in enumerate(predict_y):
            if p == 1:
                fold_pp += 1
            if test_y[index] == 1:
                fold_rp += 1
            if p == 1 and test_y[index] == 1:
                fold_tp += 1
        precision = fold_tp/fold_pp
        recall = fold_tp/fold_rp
        f_score = 2*precision*recall/(precision+recall)
        score_sum_pre += precision
        score_sum_rcl += recall
        score_sum_f += f_score
        r += 1
    return score_sum_pre/k_fold, score_sum_rcl/k_fold, score_sum_f/k_fold


# train ML based recommendation models for each category using grid search to find best parameters
# Input: directory of the train data
# Output: None
def test_ml_recommendation_grid_cv(file_path):
    c_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    k_fold = 10  # number of folds for cross validation
    data = read_data(file_path)
    # train for each individual category
    for cat in data.keys():
        print(cat, '\n')
        data_cat = data[cat]
        data_cat = data_cat.sample(frac=1).reset_index(drop=True)  # random shuffle the samples
        # print(cat, " ", len(data_cat.index))
        for c in c_list:
            score_avg_pre, score_avg_rcl, score_avg_f = test_ml_recommendation_cv(data_cat, c, k_fold)
            print(c, ' ', score_avg_pre, ' ', score_avg_rcl, ' ', score_avg_f)


""" These functions are for generating recommendation results """


# keyword recommendation for the testing file using IR based method
# Input: test file directory, sample file directory, output file directory,
#        the number of top keywords recommended,
#        the weight of click_rate in the ranking function (a dictionary - a parameter for each category)
# Output: None (the recommended keywords are stored in the output file)
def recommend_based_on_ir_method(file_path_test, file_path_sample, file_path_output, top_k, para_alpha):
    # read test file
    data = pd.read_csv(file_path_test, sep=',')
    data.columns = ['product', 'category']
    print(len(data.index), ' test records read')
    # build recommender based on sample file
    rmd = Recommender()
    rmd.load_data_from_file(file_path_sample)

    # generate recommendation for each record of the test file
    with open(file_path_output, "w", encoding='utf8') as f:
        for index, row in data.iterrows():
            alpha = para_alpha[row['category']]
            result = rmd.recommend_ir_method(row['product'], row['category'], top_k, alpha)
            result = [row['product'], row['category']] + result
            f.write(",".join(result))
            f.write('\n')


# keyword recommendation for the testing file using ML based method
# Input: test file directory, sample file directory, output file directory, top_k,
#        the regularization strength parameter (a dictionary - a parameter for each category)
# Output: None (the recommended keywords are stored in the output file)
def recommend_based_on_ml_method(file_path_test, file_path_sample, file_path_output, top_k, para_c):
    # read test file
    data_test = pd.read_csv(file_path_test, sep=',')
    data_test.columns = ['product', 'category']
    print(len(data_test.index), ' test records read')
    # read sample file
    data_train = pd.read_csv(file_path_sample, sep=',')
    data_train.columns = ['product', 'category', 'query', 'event', 'date']
    print(len(data_train.index), ' train records read')
    # build recommender based on sample data
    rmd = Recommender()
    rmd.load_data_from_df(data_train)
    # train the classifier for each category
    models = dict()
    for cat in data_train['category'].unique():
        data_train_cat = data_train.loc[data_train['category'] == cat]
        c = para_c[cat]
        models[cat] = rmd.recommend_ml_train(data_train_cat, c)

    # generate recommendation for each record of the test file
    with open(file_path_output, "w", encoding='utf8') as f:
        for index, row in data_test.iterrows():
            cat = row['category']
            result = rmd.recommend_ml_predict(row['product'], cat, top_k, models[cat])
            result = [row['product'], row['category']] + result
            f.write(",".join(result))
            f.write('\n')


def main():
    # find the best parameter setting for IR based recommendation
    # test_ir_recommendation_grid_cv("C:/Users/yzwang.SMUSTF/PycharmProjects/test/data/sample.csv")

    # find the best parameter setting for ML based recommendation
    test_ml_recommendation_grid_cv("C:/Users/yzwang.SMUSTF/PycharmProjects/test/data/sample.csv")

    # IR based recommendation
    # alpha_para = dict()
    # alpha_para['Female Fashion'] = 0.2
    # alpha_para['Male Fashion'] = 0.2
    # alpha_para['Mobile & Gadgets'] = 0.2
    # recommend_based_on_ir_method("C:/Users/yzwang.SMUSTF/PycharmProjects/test/data/test.csv",
    #                             "C:/Users/yzwang.SMUSTF/PycharmProjects/test/data/sample.csv",
    #                             "C:/Users/yzwang.SMUSTF/PycharmProjects/test/data/result_ir.csv", 5, alpha_para)

    # ML based recommendation
    # c_para = dict()
    # c_para['Female Fashion'] = 0.001
    # c_para['Male Fashion'] = 10
    # c_para['Mobile & Gadgets'] = 10
    # recommend_based_on_ml_method("C:/Users/yzwang.SMUSTF/PycharmProjects/test/data/test.csv",
    # "C:/Users/yzwang.SMUSTF/PycharmProjects/test/data/sample.csv",
    # "C:/Users/yzwang.SMUSTF/PycharmProjects/test/data/result_ml.csv", 5, c_para)


if __name__ == "__main__":
    main()
