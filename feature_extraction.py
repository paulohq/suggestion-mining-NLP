from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

class feature_extraction(object):
    def __init__(self):

        #self.CORPUS = []
        self.new_doc = []
        self.features = []
        self.feature_names = []
        self.bow_vectorizer = []
        self.bow_features = []

    def bow_extractor(self, CORPUS, ngram_range=(1, 1)):
        self.bow_vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
        self.bow_features = self.bow_vectorizer.fit_transform(CORPUS)
        self.features = self.bow_features.todense()
        self.feature_names = self.bow_vectorizer.get_feature_names()
        return self.bow_vectorizer, self.features, self.feature_names

    #def extract(self):
    #    self.bow_vectorizer, self.bow_features = extract.bow_extractor()
    #    self.features = self.bow_features.todense()
        #print (self.features)

     #   self.feature_names = self.bow_vectorizer.get_feature_names()
     #   return self.features, self.feature_names
        #print(self.feature_names)

    def display_features(self):
        df = pd.DataFrame(data=self.features, columns=self.feature_names)
        print(df)
        return df

    def new_doc_features(self):
        new_doc_features = self.bow_vectorizer.transform(self.new_doc)
        new_doc_features = new_doc_features.todense()
        return new_doc_features

extract = feature_extraction()

CORPUS = [
'the sky is blue',
'sky is blue and sky is beautiful',
'the beautiful sky is so blue',
'i love blue cheese'
]

extract.bow_extractor(CORPUS)
extract.display_features()