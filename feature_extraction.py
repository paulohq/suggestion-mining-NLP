from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
import gensim
import scipy.sparse as sp
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer

class feature_extraction(object):
    def __init__(self):

        self.model = []

    #Extract bow features for the CORPUS.
    def bow_extractor(self, CORPUS, ngram_range=(1, 1)):
        vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
        features = vectorizer.fit_transform(CORPUS)
        return vectorizer, features

    #Display features and feature names for the corpus.
    def display_features(self, features, feature_names):
        df = pd.DataFrame(data=features, columns=feature_names)
        print(df)
        return df

    #TF-IDF  features to the bow matrix passed as parameter.
    def tfidf_transformer(self, bow_matrix):
        transformer = TfidfTransformer(norm='l2',
                                       smooth_idf=True,
                                       use_idf=True)
        tfidf_matrix = transformer.fit_transform(bow_matrix)
        return transformer, tfidf_matrix

    #Extract TF-IDF for the new document using built tfidf transformer.
    def tfidf_new_doc_features(self, new_doc, bow_vectorizer, tfidf_trans):
        new_doc_features = bow_vectorizer.transform(new_doc)
        nd_tfidf = tfidf_trans.transform(new_doc_features)
        nd_features = np.round(nd_tfidf.todense(), 2)
        feature_names = bow_vectorizer.get_feature_names()
        return nd_features, feature_names

    #Extract bow features for new document in the test.
    def bow_new_doc_features(self, bow_vectorizer, new_doc):
        new_doc_features = bow_vectorizer.transform(new_doc)
        new_doc_features = new_doc_features.todense()
        return new_doc_features

    #compute the tfidf-based feature vectors for documents
    def tfidf_extractor(self, corpus, ngram_range=(1, 1)):
        vectorizer = TfidfVectorizer(min_df=1,
                                     norm='l2',
                                     smooth_idf=True,
                                     use_idf=True,
                                     ngram_range=ngram_range)
        features = vectorizer.fit_transform(corpus)
        return vectorizer, features

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
    #to perform averaging of word vectors for a corpus of documents
    def averaged_word_vectorizer(self, corpus, model, num_features):
        vocabulary = set(model.wv.index2word)
        features = [self.average_word_vectors(tokenized_sentence, model, vocabulary,num_features)
                                    for tokenized_sentence in corpus]
        return np.array(features)

    #Create the model word2vec.
    def create_model_word2vec(self, TOKENIZED_CORPUS, size, window, min_count, sample):
        model = gensim.models.Word2Vec(TOKENIZED_CORPUS, size=size, window=window, min_count=min_count, sample=sample)

        return model

    # define function to compute tfidf weighted averaged word vector for a document
    def tfidf_wtd_avg_word_vectors(self, words, tfidf_vector, tfidf_vocabulary, model, num_features):
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

    # generalize above function for a corpus of documents.
    #Created to perform TF-IDF weighted averaging of word vectors for a corpus of documents.
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
'''
#bow extrator
bow_vectorizer, bow_features = extract.bow_extractor(CORPUS)
features = bow_features.todense()
print(features)

new_doc_features = bow_vectorizer.transform(new_doc)
new_doc_features = new_doc_features.todense()
print (new_doc_features)

feature_names = bow_vectorizer.get_feature_names()
print (feature_names)
extract.display_features(features, feature_names)
extract.display_features(new_doc_features, feature_names)

#tf-idf transform
print('tf-idf extractor')
tfidf_trans, tdidf_features = extract.tfidf_transformer(bow_features)
features = np.round(tdidf_features.todense(), 2)
extract.display_features(features, feature_names)

# show tfidf features for new_doc using built tfidf transformer
print('tf-idf new doc')
nd_tfidf = tfidf_trans.transform(new_doc_features)
nd_features = np.round(nd_tfidf.todense(), 2)
extract.display_features(nd_features, feature_names)

#tf-idf advanced
# compute term frequency
print('tf-idf advanced')
print('tf')
tf = bow_features.todense()
tf = np.array(tf, dtype='float64')
#show term frequencies
extract.display_features(tf, feature_names)
# build the document frequency matrix
print('df')
df = np.diff(sp.csc_matrix(bow_features, copy=True).indptr)
df = 1 + df # to smoothen idf later
# show document frequencies
extract.display_features([df], feature_names)

# compute inverse document frequencies
print('idf')
total_docs = 1 + len(CORPUS)
idf = 1.0 + np.log(float(total_docs) / df)
# show inverse document frequencies
extract.display_features([np.round(idf, 2)], feature_names)

# compute idf diagonal matrix
print('idf diagonal matrix')
total_features = bow_features.shape[1]
idf_diag = sp.spdiags(idf, diags=0, m=total_features, n=total_features)
idf = idf_diag.todense()
# print the idf diagonal matrix
print (np.round(idf, 2))

# compute tfidf feature matrix
print('tfidf matrix')
tfidf = tf * idf
# show tfidf feature matrix
extract.display_features(np.round(tfidf, 2), feature_names)

# compute L2 norms
norms = norm(tfidf, axis=1)
# print norms for each document
print('L2 Norm')
print (np.round(norms, 2))
# compute normalized tfidf
norm_tfidf = tfidf / norms[:, None]
# show final tfidf feature matrix
print('final Tf-idf matrix')
extract.display_features(np.round(norm_tfidf, 2), feature_names)

# build tfidf vectorizer and get training corpus feature vectors
print('TF-IDF directly::')
tfidf_vectorizer, tfidf_features = extract.tfidf_extractor(CORPUS)
extract.display_features(np.round(tfidf_features.todense(), 2), feature_names)
# get tfidf feature vector for the new document
nd_tfidf = tfidf_vectorizer.transform(new_doc)
extract.display_features(np.round(nd_tfidf.todense(), 2), feature_names)

#extract.tfid_extractor(CORPUS)
#extract.display_features(features, feature_names)
#################
'''



# tokenize corpora
TOKENIZED_CORPUS = [nltk.word_tokenize(sentence)
                    for sentence in CORPUS]
tokenized_new_doc = [nltk.word_tokenize(sentence)
                    for sentence in new_doc]
# build the word2vec model on our training corpus
model = extract.create_model_word2vec(TOKENIZED_CORPUS, size=10, window=10, min_count=2, sample=1e-3)

# get averaged word vectors for our training CORPUS
print('word2vec')
avg_word_vec_features = extract.averaged_word_vectorizer(corpus=TOKENIZED_CORPUS, model=model, num_features=10)
print (np.round(avg_word_vec_features, 3))
avg_word_vec_features_new_doc = extract.averaged_word_vectorizer(corpus=tokenized_new_doc, model=model, num_features=10)
print (np.round(avg_word_vec_features_new_doc, 3))
print(model['sky'])


print('tfidf_weighted_averaged:')
tfidf_vectorizer, tfidf_features = extract.tfidf_extractor(CORPUS)
#get tfidf weights and vocabulary from earlier results and compute result
corpus_tfidf = tfidf_features
vocab = tfidf_vectorizer.vocabulary_
wt_tfidf_word_vec_features = extract.tfidf_weighted_averaged_word_vectorizer(corpus=TOKENIZED_CORPUS, tfidf_vectors=corpus_tfidf
                                                                             ,tfidf_vocabulary=vocab, model=model,num_features=10)
print (np.round(wt_tfidf_word_vec_features, 3))

#nd_tfidf, nd_features = extract.tfidf_new_doc_features(tokenized_new_doc)
# compute avgd word vector for test new_doc
#nd_wt_tfidf_word_vec_features = extract.tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_new_doc, tfidf_vectors=nd_tfidf
#                                                                        , tfidf_vocabulary=vocab, model=model, num_features=10)
#print (np.round(nd_wt_tfidf_word_vec_features, 3))
#print(model['blue'])


