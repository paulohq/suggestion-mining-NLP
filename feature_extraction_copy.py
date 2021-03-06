from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
import gensim

class feature_extraction(object):
    def __init__(self):

        #self.CORPUS = []
        self.new_doc = []
        self.features = []
        self.feature_names = []
        self.bow_vectorizer = []
        self.bow_features = []
        self.tfidf_trans = []
        self.tfidf_features = []

        self.model = []

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

    def display_features(self, features, feature_names):
        df = pd.DataFrame(data=features, columns=feature_names)
        print(df)
        return df

    def tfidf_transformer(self, bow_matrix):
        transformer = TfidfTransformer(norm='l2',
                                       smooth_idf=True,
                                       use_idf=True)
        tfidf_matrix = transformer.fit_transform(bow_matrix)
        return transformer, tfidf_matrix

    #Extract the TF-IDF for the corpus.
    def tfid_extractor(self, CORPUS, ngram_range=(1, 1)):
        self.bow_vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
        self.bow_features = self.bow_vectorizer.fit_transform(CORPUS)
        self.feature_names = self.bow_vectorizer.get_feature_names()

        self.tfidf_trans, self.tfidf_features = self.tfidf_transformer(self.bow_features)
        self.features = np.round(self.tfidf_features.todense(), 2)

        return self.bow_vectorizer, self.features, self.feature_names

    #Extract TF-IDF for the new document using built tfidf transformer.
    def tfidf_new_doc_features(self, new_doc):
        new_doc_features = self.bow_vectorizer.transform(new_doc)
        nd_tfidf = self.tfidf_trans.transform(new_doc_features)
        nd_features = np.round(nd_tfidf.todense(), 2)

        return nd_features

    def bow_new_doc_features(self):
        new_doc_features = self.bow_vectorizer.transform(self.new_doc)
        new_doc_features = new_doc_features.todense()
        return new_doc_features

    # define function to average word vectors for a text document
    def average_word_vectors(self, words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        for word in words:
            if word in vocabulary:
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model[word])

        if nwords:
            feature_vector = np.divide(feature_vector, nwords)
        return feature_vector

    # generalize above function for a corpus of documents.
    def averaged_word_vectorizer(self, corpus, model, num_features):
        vocabulary = set(model.wv.index2word)
        features = [self.average_word_vectors(tokenized_sentence, model, vocabulary,
                                          num_features)
                     for tokenized_sentence in corpus]
        return np.array(features)

    #Create the model word2vec.
    def create_model_word2vec(self, TOKENIZED_CORPUS, size, window, min_count, sample):
        self.model = gensim.models.Word2Vec(TOKENIZED_CORPUS, size=size, window=window, min_count=min_count, sample=sample)
        # get averaged word vectors for our training CORPUS
        avg_word_vec_features = self.averaged_word_vectorizer(corpus=TOKENIZED_CORPUS, model=self.model, num_features=10)
        print(np.round(avg_word_vec_features, 3))

    # define function to compute tfidf weighted averaged word vector for a document
    def tfidf_wtd_avg_word_vectors(self,words, tfidf_vector, tfidf_vocabulary, model, num_features):
        word_tfidfs = [tfidf_vector[0, tfidf_vocabulary.get(word)]
                       if tfidf_vocabulary.get(word)
                       else 0 for word in words]

        word_tfidf_map = {word: tfidf_val for word, tfidf_val in zip(words, word_tfidfs)}
        feature_vector = np.zeros((num_features,), dtype="float64")
        vocabulary = set(model.wv.index2word)
        wts = 0.
        for word in words:
            if word in vocabulary:
                word_vector = model[word]
                weighted_word_vector = word_tfidf_map[word] * word_vector
                wts = wts + word_tfidf_map[word]
                feature_vector = np.add(feature_vector, weighted_word_vector)
        if wts:
            feature_vector = np.divide(feature_vector, wts)
        return feature_vector

    # generalize above function for a corpus of documents
    def tfidf_weighted_averaged_word_vectorizer(self, corpus, tfidf_vectors, tfidf_vocabulary, model, num_features):
        docs_tfidfs = [(doc, doc_tfidf)
                       for doc, doc_tfidf
                       in zip(corpus, tfidf_vectors)]
        features = [self.tfidf_wtd_avg_word_vectors(tokenized_sentence, tfidf, tfidf_vocabulary, model, num_features)
                            for tokenized_sentence, tfidf in docs_tfidfs]
        return np.array(features)

extract = feature_extraction()

CORPUS = [
'the sky is blue',
'sky is blue and sky is beautiful',
'the beautiful sky is so blue',
'i love blue cheese'
]

new_doc = ['loving this blue sky today']

# tokenize corpora
TOKENIZED_CORPUS = [nltk.word_tokenize(sentence)
                    for sentence in CORPUS]
tokenized_new_doc = [nltk.word_tokenize(sentence)
                    for sentence in new_doc]
# build the word2vec model on our training corpus
model = gensim.models.Word2Vec(TOKENIZED_CORPUS, size=10, window=10, min_count=2, sample=1e-3)

tfidf_vectorizer, tfidf_features, tfidf_feature_names = extract.tfid_extractor(CORPUS)
# get tfidf weights and vocabulary from earlier results and compute result
corpus_tfidf = tfidf_features
vocab = tfidf_vectorizer.vocabulary_
wt_tfidf_word_vec_features = extract.tfidf_weighted_averaged_word_vectorizer(corpus=TOKENIZED_CORPUS, tfidf_vectors=corpus_tfidf
                                                                             ,tfidf_vocabulary=vocab, model=model,num_features=10)
print (np.round(wt_tfidf_word_vec_features, 3))

nd_tfidf, nd_features = extract.tfidf_new_doc_features(tokenized_new_doc)
# compute avgd word vector for test new_doc
nd_wt_tfidf_word_vec_features = extract.tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_new_doc, tfidf_vectors=nd_tfidf
                                                                        , tfidf_vocabulary=vocab, model=model, num_features=10)
print (np.round(nd_wt_tfidf_word_vec_features, 3))
#print(model['blue'])

# get averaged word vectors for our training CORPUS
#avg_word_vec_features = extract.averaged_word_vectorizer(corpus=TOKENIZED_CORPUS, model=model, num_features=10)
#print (np.round(avg_word_vec_features, 3))

#extract.tfid_extractor(CORPUS)
#extract.display_features(self.features, self.feature_names)

