import jieba
import re
import operator
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from recommender.keyword_index import KeywordIndex
from recommender.inverted_index import InvertedIndex


class Recommender:
    """ The recommender class recommends top_k keywords for a given product title
        Notes:
        1. Must build the indexes from sample file using the load_data_from_file() function
        before run the recommendation() function to recommend keywords. Otherwise, no keywords will be returned
        2. The index is build incrementally from the data of the input file
    """
    def __init__(self):
        self.keyword_index_cats = dict()
        self.inverted_index_cats = dict()

    """ --- utility functions --- """
    # Remove none word/number characters and extra whitespaces in a text string
    # Input: a string; Output: a cleaned string without special characters
    @staticmethod
    def strip(line):
        line = line.lower()
        line = re.sub(r"[^\w\s]", ' ', line)
        line = re.sub(r"\s+", ' ', line)
        return line

    # Tokenize a text string (using jieba package)
    # Input: a string; Output: a list of tokens
    @staticmethod
    def get_token(line):
        line = Recommender.strip(line)
        # cut the string for more tokens (good for search)
        seg_list = jieba.lcut_for_search(line)
        # remove empty tokens with only whitespace
        seg_list = [x for x in seg_list if x not in [' ', '']]
        return seg_list

    # Obtain the click rate of the given keywords (Used for testing)
    # Input: a list of keywords, category; Output: dictionary of keywords and their click rate
    def get_click_rate(self, keyword_list, category):
        return self.keyword_index_cats[category].query_click_rate(keyword_list)

    # Load data from given file and build related indexes incrementally (a set of indexes for each category)
    # Input: the directory and name of the data file; Output: None
    def load_data_from_file(self, file_path):
        try:
            with open(file_path, encoding='utf8') as file:
                lcount = 0
                for cnt, line in enumerate(file):
                    if cnt > 0:  # skip the first line of the file
                        title, category, keyword, event, date = line.split(',')
                        # build indexes for each category of products
                        if category not in self.keyword_index_cats.keys():
                            self.keyword_index_cats[category] = KeywordIndex()
                        if category not in self.inverted_index_cats.keys():
                            self.inverted_index_cats[category] = InvertedIndex()
                        keyword = self.strip(keyword)  # remove special charactors in the query keywords
                        keyword_list = keyword.split(' ')
                        token_list = self.get_token(title)
                        for kw in keyword_list:
                            self.keyword_index_cats[category].insert(kw, event)
                            self.inverted_index_cats[category].insert(kw, token_list)
                        lcount += 1
            # print(lcount, " records are read to build index")
        except IOError:
            print("Could not read file ", file_path)

    # Load data from dataframe and build related indexes incrementally (a set of indexes for each category)
    # Input: the dataframe; Output: None
    def load_data_from_df(self, data):
        lcount = 0
        for index, row in data.iterrows():
            title = row['product']
            category = row['category']
            keyword = row['query']
            event = row['event']
            # build indexes for each category of products
            if category not in self.keyword_index_cats.keys():
                self.keyword_index_cats[category] = KeywordIndex()
            if category not in self.inverted_index_cats.keys():
                self.inverted_index_cats[category] = InvertedIndex()
            keyword = self.strip(keyword)  # remove special characters in the query keywords
            keyword_list = keyword.split(' ')
            token_list = self.get_token(title)
            for kw in keyword_list:
                self.keyword_index_cats[category].insert(kw, event)
                self.inverted_index_cats[category].insert(kw, token_list)
            lcount += 1
        # print(lcount, " records are read to build index")

    """ --- IR based recommendation --- """
    # Recommend top_k keywords of a a given product
    # Input: product title, product category, top_k, alpha (the tuneable weight of the click rate);
    # Output: a list of k keywords
    # Note: It is possible the function returns less than k keywords
    #       if it does not find enough relevant keywords of the product
    def recommend_ir_method(self, title, category, k, alph):
        token_list = self.get_token(title)
        # retrive relevant keywords to the product with relevance score
        relevent_keywords = self.inverted_index_cats[category].query(token_list)
        # retrive the click rate of the relevent keywords
        click_rate = self.keyword_index_cats[category].query_click_rate(relevent_keywords.keys())

        kscore = dict()  # the final score of the keywords
        for key, rscore in relevent_keywords.items():
            kscore[key] = (1-alph)*rscore+alph*click_rate[key]
        # sort the keywords based on the final score
        sorted_keywords = sorted(kscore.items(), key=operator.itemgetter(1), reverse=True)
        result = [x[0] for x in sorted_keywords]
        return result[0:k]

    """ --- ML based recommendation --- """
    # extract features of each log event record for classification
    # Input: product category, product title, query;
    # Output: a feature vector including relevance score, impression #, click #, click rate
    def get_feature(self, category, title, query):
        token_list = self.get_token(title)
        keywords = self.strip(query).split(' ')
        cosin_score = self.inverted_index_cats[category].get_cosin_score(token_list, keywords)
        cosin_score = sum(cosin_score.values()) / len(keywords)
        impression = self.keyword_index_cats[category].query_frequency(keywords, 'Impression')
        impression = sum(impression.values()) / len(keywords)
        click = self.keyword_index_cats[category].query_frequency(keywords, 'Click')
        click = sum(click.values()) / len(keywords)
        # form a feature vector
        if (click + impression) > 0:
            x = [cosin_score, impression, click, click / (click + impression)]
        else:
            x = [cosin_score, impression, click, 0]
        return x

    # training classification model for event classifier
    # Input: the training data, regularization strength parameter
    # Output: F-score, a trained model
    def recommend_ml_train(self, data_train, para_c):
        # feature extraction
        data_x = np.empty([0, 4])  # features
        data_y = np.empty([0, 1])  # class labels
        for index, row in data_train.iterrows():
            x = self.get_feature(row['category'], row['product'], row['query'])
            data_x = np.append(data_x, [x], axis=0)
            # click event as positive example, no-click (impression) event as negative example
            if row['event'] == 'Click':
                data_y = np.append(data_y, [[1]], axis=0)
            else:
                data_y = np.append(data_y, [[0]], axis=0)

        # transform features and fit the logistic model
        clf = Pipeline(steps=[
            # ('encoder', OneHotEncoder(categorical_features=[0])),  # categorical feature encoder
            ('standerizer', StandardScaler()),  # feature standardization
            # logistic regression with default l2 regularization and unbalanced classes
            ('regression', LogisticRegression(class_weight='balanced', max_iter=500, C=para_c))
        ])
        clf.fit(data_x, data_y.ravel())
        return clf

    # recommend keywords to a product using the trained classification model
    # first locate relevent keywords, then, recommend the top_k keywords that are classified as positive click keyword
    # Input: a product and its category, top_k, the trained model
    # Output: a list of recommended keywords
    def recommend_ml_predict(self, title, category, k, model):
        result = []
        # search for relevant keywords of the given product
        token_list = self.get_token(title)
        keyword_list = self.inverted_index_cats[category].query(token_list)
        # for each relevant keywords, used the model to predict if there will be a click
        kscore = dict()  # the final score of the keywords
        for key in keyword_list:
            x = self.get_feature(category, title, key)
            score = model.predict_proba([x])  # get the predict score of a keyword
            if score[0,1]>score[0,0]:
                kscore[key] = score[0,1]
        sorted_keywords = sorted(kscore.items(), key=operator.itemgetter(1), reverse=True)
        result = [x[0] for x in sorted_keywords]
        return result[0:k]
