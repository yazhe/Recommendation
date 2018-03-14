from itertools import groupby
import math
from scipy import spatial


class InvertedIndex:
    """ The InvertedIndex class indexes the query keywords by the terms of the product titles. It is used to
    calculate the tf-idf similarity score of a product to query keywords
            Notes:
            1. The index is build incrementally by adding records one by one using the insert() function
            2. The class provides various query functions to query cosine similarity score to a set of query keywords
            or all query keywords
    """
    def __init__(self):
        # a list of all the tokens related to a query keyword
        # (cumulated from all the product titles related to the keyword)
        self.kset = dict()
        # tokens' inverted index to the related keywords
        self.ivindex = dict()

    # get the frequency of the tokens in a token list
    # Input: a list of tokens; Output: the frequency of each individual token of the list
    @staticmethod
    def get_tf(token_list):
        tfreq = dict()
        token_list.sort()
        for key, group in groupby(token_list):
            tfreq[key] = len(list(group))
        return tfreq

    # insert a record to the inverted index
    # Input: keyword, a list of tokens; Output: None
    def insert(self, keyword, token_list):
        tfreq = InvertedIndex.get_tf(token_list)
        if keyword not in self.kset.keys():
            self.kset[keyword] = token_list
        else:
            self.kset[keyword] = self.kset[keyword] + token_list
        for t, v in tfreq.items():
            if t not in self.ivindex.keys():
                self.ivindex[t] = dict()
                self.ivindex[t][keyword] = v
            elif keyword not in self.ivindex[t].keys():
                self.ivindex[t][keyword] = v
            else:
                self.ivindex[t][keyword] += v

    # calculate the cosine similarity score of a token list to the individual keyword of a given keyword list
    # Input: the token list and a list of keywords; Output: the cosine score to each of the keyword
    def get_cosin_score(self, token_list, keywords):
        result = dict()
        qfreq = InvertedIndex.get_tf(token_list)
        qvec = []  # the query vector based on tf-idf
        for t in qfreq.keys():
            # normalized term frequency of a query term
            norm_tf = qfreq[t] / len(token_list)
            idf = 0
            if t in self.ivindex.keys():
                idf = math.log(len(self.kset) / len(self.ivindex[t]))  # idf score
            qvec.append(norm_tf * idf)
        for key in keywords:
            kvec = []  # the keyword vector based on tf-idf
            for t in qfreq.keys():
                if (t in self.ivindex.keys()) and (key in self.ivindex[t].keys()):
                    norm_tf = self.ivindex[t][key] / len(self.kset[key])
                    idf = math.log(len(self.kset) / len(self.ivindex[t]))
                    kvec.append(norm_tf * idf)
                else:
                    kvec.append(0)
            if sum(kvec) != 0:
                score = 1 - spatial.distance.cosine(qvec, kvec)
                result[key] = score
            else:
                result[key] = 0
        return result

    # calculate the cosine similarity score of a token list to every related keyword in the system
    # Input: the token list and a list of keywords; Output: the cosine score to each of the related keyword
    def query(self, token_list):
        result = dict()
        qfreq = self.get_tf(token_list)
        qvec = []
        keyword_set = set()
        for t in qfreq.keys():
            # normalized term frequency of a query term
            norm_tf = qfreq[t]/len(token_list)
            idf = 0
            if t in self.ivindex.keys():
                idf = math.log(len(self.kset)/len(self.ivindex[t]))
                keyword_set |= set(self.ivindex[t].keys())
            qvec.append(norm_tf*idf)

        for k in keyword_set:
            kvec = []
            for t in qfreq.keys():
                if (t in self.ivindex.keys()) and (k in self.ivindex[t].keys()):
                    norm_tf = self.ivindex[t][k]/len(self.kset[k])
                    idf = math.log(len(self.kset)/len(self.ivindex[t]))
                    kvec.append(norm_tf*idf)
                else:
                    kvec.append(0)
            score = 1-spatial.distance.cosine(qvec, kvec)
            result[k] = score
        return result
