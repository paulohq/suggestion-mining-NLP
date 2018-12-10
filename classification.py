from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from normalization import normalize_corpus

from feature_extraction import *
import nltk
import gensim

from sklearn import metrics
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier




def get_data():
    data = fetch_20newsgroups(subset='all',
                              shuffle=True,
                              remove=('headers', 'footers', 'quotes'))
    return data

def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels,
                                                        test_size=0.33, random_state=42)
    return train_X, test_X, train_Y, test_Y

def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)
    return filtered_corpus, filtered_labels


def get_metrics(true_labels, predicted_labels):
    print
    'Accuracy:', np.round(
        metrics.accuracy_score(true_labels,
                               predicted_labels), 2)
    print
    'Precision:', np.round(
        metrics.precision_score(true_labels,
                                predicted_labels,
                                average='weighted'), 2)
    print
    'Recall:', np.round(
        metrics.recall_score(true_labels,
                             predicted_labels,
                             average='weighted'), 2)
    print
    'F1 Score:', np.round(
        metrics.f1_score(true_labels,
                         predicted_labels,
                         average='weighted'), 2)


def train_predict_evaluate_model(classifier,
                                 train_features, train_labels,
                                 test_features, test_labels):
    # build model
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features)
    # evaluate model prediction performance
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)
    return predictions

dataset = get_data()
print (dataset.target_names)
corpus, labels = dataset.data, dataset.target
corpus, labels = remove_empty_docs(corpus, labels)

print ('Sample document:', corpus[10])
print ('Class label:',labels[10])
print ('Actual class label:', dataset.target_names[labels[10]])

# prepare train and test datasets
train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus, labels, test_data_proportion=0.3)

norm_train_corpus = normalize_corpus(train_corpus)
norm_test_corpus = normalize_corpus(test_corpus)

extract = feature_extraction()
# bag of words features
bow_vectorizer, bow_train_features = extract.bow_extractor(norm_train_corpus)
bow_test_features = bow_vectorizer.transform(norm_test_corpus)

# tfidf features
tfidf_vectorizer, tfidf_train_features = extract.tfidf_extractor(norm_train_corpus)
tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)
# tokenize documents
tokenized_train = [nltk.word_tokenize(text)
                   for text in norm_train_corpus]
tokenized_test = [nltk.word_tokenize(text)
                   for text in norm_test_corpus]
# build word2vec model
model = gensim.models.Word2Vec(tokenized_train,
                               size=500,
                               window=100,
                               min_count=30,
                               sample=1e-3)
# averaged word vector features
avg_wv_train_features = extract.averaged_word_vectorizer(corpus=tokenized_train,
                                                 model=model,
                                                 num_features=500)
avg_wv_test_features = extract.averaged_word_vectorizer(corpus=tokenized_test,
                                                model=model,
                                                num_features=500)

# tfidf weighted averaged word vector features
vocab = tfidf_vectorizer.vocabulary_
tfidf_wv_train_features = extract.tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_train, tfidf_vectors= tfidf_train_features,
                                                                  tfidf_vocabulary=vocab, model=model,num_features=500)
tfidf_wv_test_features = extract.tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_test,tfidf_vectors=tfidf_test_features,
                                                                  tfidf_vocabulary=vocab, model=model,num_features=500)


mnb = MultinomialNB()
svm = SGDClassifier(loss='hinge', n_iter=100)

# Multinomial Naive Bayes with bag of words features
mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb, train_features=bow_train_features, train_labels=train_labels,
                                                   test_features=bow_test_features, test_labels=test_labels)

# Support Vector Machine with bag of words features
svm_bow_predictions = train_predict_evaluate_model(classifier=svm,train_features=bow_train_features,train_labels=train_labels,
                                                             test_features=bow_test_features,test_labels=test_labels)

# Multinomial Naive Bayes with tfidf features
mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,train_features=tfidf_train_features, train_labels=train_labels,
                                                     test_features=tfidf_test_features,test_labels=test_labels)

# Support Vector Machine with tfidf features
svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,train_features=tfidf_train_features,train_labels=train_labels,
                                                    test_features=tfidf_test_features,test_labels=test_labels)

# Support Vector Machine with averaged word vector features
svm_avgwv_predictions = train_predict_evaluate_model(classifier=svm,train_features=avg_wv_train_features,train_labels=train_labels,
                                                    test_features=avg_wv_test_features,test_labels=test_labels)


# Support Vector Machine with tfidf weighted averaged word vector features
svm_tfidfwv_predictions = train_predict_evaluate_model(classifier=svm,train_features=tfidf_wv_train_features,
                                                      train_labels=train_labels, test_features=tfidf_wv_test_features,test_labels=test_labels)
