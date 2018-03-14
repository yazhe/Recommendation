class KeywordIndex:
    """ The KeyWordIndex class records the keywords and their impression # and click # in the dataset
        Notes:
        1. The index is build incrementally by adding records one by one using the insert() function
        2. The class provides various query function to query the impression #, click #, and click rate
        of given keywords
    """
    def __init__(self):
        self.kindex = dict()
        # a small value is returned in case a keyword has zero click
        self.small_value = 0.00001

    # insert one record to the index
    # Input: a keyword and event; Output: None
    def insert(self, keyword, event):
        if keyword not in self.kindex.keys():
            self.kindex[keyword] = dict()
            self.kindex[keyword]['Impression'] = 0
            self.kindex[keyword]['Click'] = 0
        self.kindex[keyword][event] += 1

    # calculate the click rate of given keywords
    # Input: a list of keywords; Output: a dictionary of keyword and click rate
    def query_click_rate(self, keyword_list):
        result = dict()
        for key in keyword_list:
            if self.kindex[key]['Click'] == 0:
                rate = self.small_value
            else:
                rate = self.kindex[key]['Click']/(self.kindex[key]['Click']+self.kindex[key]['Impression'])
            result[key] = rate
        return result

    # calculate the frequency of Impression or Click of given keywords
    # Input: a list of keywords, queried event (Impression/Click);
    # Output: a dictionary of keyword and corresponding frequency
    def query_frequency(self, keyword_list, event):
        result = dict()
        if event in ['Impression', 'Click']:
            for key in keyword_list:
                if key in self.kindex.keys():
                    rate = self.kindex[key][event]
                    result[key] = rate
                else:
                    result[key] = 0
        else:
            print("please input an valid event (Impression or Click)")
        return result

    # print the keyword index (used for testing)
    def show(self):
        for key in self.kindex.keys():
            print(key, ',', self.kindex[key]['Impression'], ',', self.kindex[key]['Click'])
