# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
from feature_extraction import *
from pre_processing import *
from read_csv import *
from contractions import contractions_dict
from classification import *

class text_classification(object):
    def __init__(self):

        self.filtered_list_expand_contractions = []
        self.train_list = []
        self.test_list = []
        self.token_list = []
        self.filtered_list = []
        self.filtered_list_lowercase = []
        self.filtered_list_remove_stopwords = []
        self.filtered_list_remove_repeated_characters = []
        self.list_spell_checker = []
        self.filtered_list_stemmer = []
        self.filtered_list_lemma = []

        self.ext = []




classifier = text_classification()
read = read_csv()
clas = classification()
proc = pre_processing()

#Read the train and test corpus.
classifier.train_list = read.read_csv("training-full-v13-bkp.csv")
classifier.test_list = read.read_csv("TrialData_SubtaskA_Test.csv",True)

#Separeted the sentences and labels from the train corpus.
train_corpus = []
train_labels = []
for text in classifier.train_list:
    train_corpus.append(text[1])
    train_labels.append(text[2])

#Separeted the sentences and labels from the test corpus.
test_corpus = []
test_labels = []
for text in classifier.test_list:
    test_corpus.append(text[1])
    test_labels.append('0')

norm_train_corpus = proc.normalize_corpus(train_corpus)
norm_test_corpus = proc.normalize_corpus(test_corpus)


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
mnb_bow_predictions = clas.train_predict_evaluate_model(classifier=mnb, train_features=bow_train_features, train_labels=train_labels,
                                                   test_features=bow_test_features, test_labels=test_labels)
print(mnb_bow_predictions)
label_mnb = []
for label in mnb_bow_predictions:
    label_mnb.append(label)

out_path = "TrialData_SubtaskA_Test_MNB_predictions.csv"
read.write_csv(classifier.test_list, label_mnb, out_path)

# Support Vector Machine with bag of words features
svm_bow_predictions = clas.train_predict_evaluate_model(classifier=svm,train_features=bow_train_features,train_labels=train_labels,
                                                             test_features=bow_test_features,test_labels=test_labels)
print (svm_bow_predictions)

label_svm = []
for label in svm_bow_predictions:
    label_svm.append(label)

out_path = "TrialData_SubtaskA_Test_SVM_predictions.csv"
read.write_csv(classifier.test_list, label_svm, out_path)

# Multinomial Naive Bayes with tfidf features
mnb_tfidf_predictions = clas.train_predict_evaluate_model(classifier=mnb,train_features=tfidf_train_features, train_labels=train_labels,
                                                     test_features=tfidf_test_features,test_labels=test_labels)
print(mnb_tfidf_predictions)
label_mnb_tfidf = []
for label in mnb_tfidf_predictions:
    label_mnb_tfidf.append(label)

out_path = "TrialData_SubtaskA_Test_MNB_TFIDF_predictions.csv"
read.write_csv(classifier.test_list, label_mnb_tfidf, out_path)

# Support Vector Machine with tfidf features
svm_tfidf_predictions = clas.train_predict_evaluate_model(classifier=svm,train_features=tfidf_train_features,train_labels=train_labels,
                                                    test_features=tfidf_test_features,test_labels=test_labels)
print(svm_tfidf_predictions)
label_svm_tfidf = []
for label in svm_tfidf_predictions:
    label_svm_tfidf.append(label)

out_path = "TrialData_SubtaskA_Test_SVM_TFIDF_predictions.csv"
read.write_csv(classifier.test_list, label_svm_tfidf, out_path)

# Support Vector Machine with averaged word vector features
svm_avgwv_predictions = clas.train_predict_evaluate_model(classifier=svm,train_features=avg_wv_train_features,train_labels=train_labels,
                                                    test_features=avg_wv_test_features,test_labels=test_labels)
print(svm_avgwv_predictions)
label_svm_avwv = []
for label in svm_avgwv_predictions:
    label_svm_avwv.append(label)

out_path = "TrialData_SubtaskA_Test_SVM_AVWV_predictions.csv"
read.write_csv(classifier.test_list, label_svm_avwv, out_path)


# Support Vector Machine with tfidf weighted averaged word vector features
svm_tfidfwv_predictions = clas.train_predict_evaluate_model(classifier=svm,train_features=tfidf_wv_train_features,
                                                      train_labels=train_labels, test_features=tfidf_wv_test_features,test_labels=test_labels)
print(svm_tfidfwv_predictions)
label_svm_tfidf_wv = []
for label in svm_tfidfwv_predictions:
    label_svm_tfidf_wv.append(label)

out_path = "TrialData_SubtaskA_Test_SVM_TFIDF_WV_predictions.csv"
read.write_csv(classifier.test_list, label_svm_tfidf_wv, out_path)

#cm = metrics.confusion_matrix(test_labels, svm_tfidf_predictions)
#print(pd.DataFrame(cm, index=range(0,20), columns=range(0,20)))

'''
#class_names = dataset.target_names
print (class_names[0], '->', class_names[15])
print (class_names[18], '->', class_names[16] )
print (class_names[19], '->', class_names[15])

num = 0
for document, label, predicted_label in zip(test_corpus, test_labels, svm_tfidf_predictions):
    if label == 0 and predicted_label == 15:
        print ('Actual Label:', class_names[label])
        print ('Predicted Label:', class_names[predicted_label])
        print ('Document:-')
        print (re.sub('\n', ' ', document))
        print ('')
        num += 1
        if num == 4:
            break
'''

############################
#sent_tokenize(sent_list)
#word_tokenize(sent_list)

#print(classifier.sent_list[0][1])
#for text in classifier.sent_list:
#    print(text[1])



'''
classifier.filtered_list_expand_contractions = [pre.expand_contractions(text[1], contractions_dict) for text in classifier.train_list]
print("Expand contractions:")
print(classifier.filtered_list_expand_contractions)

classifier.token_list = [pre.word_tokenize(sent) for sent in classifier.filtered_list_expand_contractions]
#Print list after execute word tokenize.
#pprint(token_list)

#Loop to remove special characters from the list token_list.
for sentence_tokens in classifier.token_list:
    for tokens in sentence_tokens:
        classifier.filtered_list.append(list(filter(None, [pre.remove_characters_after_tokenization(tokens)])))
#Print list after remove special characters.
#print(filtered_list)

#Loop to convert words in filtered_list to lowercase.
for sentence_tokens in classifier.filtered_list:
    for tokens in sentence_tokens:
        classifier.filtered_list_lowercase.append(list(filter(None, [pre.lower_case(tokens)])))
#Print list after convert to lowercase.
#print(filtered_list_lowercase)

#Loop to remove the stopwords from the list filtered_list_lowercase
for sentence_tokens in classifier.filtered_list_lowercase:
    for tokens in sentence_tokens:
        classifier.filtered_list_remove_stopwords.append(list(filter(None, [pre.remove_stopwords(tokens)])))

#Print list after remove stopwords
#print (filtered_list_remove_stopwords)

#Loop to remove the repeated characters from the list filtered_list_remove_stopwords.
for sentence_tokens in classifier.filtered_list_remove_stopwords:
    for tokens in sentence_tokens:
        classifier.filtered_list_remove_repeated_characters.append(list(filter(None, [pre.remove_repeated_characters(tokens)])))
#Print list after remove repeated characters.
#for sentence_tokens in classifier.filtered_list_remove_repeated_characters:
#    for tokens in sentence_tokens:
#      print(tokens)

# Loop to spell checker.
#for sentence_tokens in classifier.filtered_list_remove_repeated_characters:
#    for tokens in sentence_tokens:
#        classifier.list_spell_checker.append((list(filter(None, [pre.correction(tokens)]))))
#print('Speel checker:')
#print(classifier.list_spell_checker)

#Loop to stemming.
#for sentence_tokens in classifier.filtered_list_remove_repeated_characters:
#    for tokens in sentence_tokens:
#        classifier.filtered_list_stemmer.append(list(filter(None, [pre.lancaster_stemmer(tokens)])))
#Print list after stemming.
#print(filtered_list_stemmer)


#Loop to lemmatize
for sentence_tokens in classifier.filtered_list_remove_repeated_characters:
    for tokens in sentence_tokens:
        classifier.filtered_list_lemma.append(list(filter(None, [pre.lemmatizer(tokens)])))
print('Lemma:')
print(classifier.filtered_list_lemma)

text = ""
corpus = []
for reg in classifier.filtered_list_lemma:
    for r in reg:
        for i in r:
            text = text + ' ' + i
        corpus.append(text)
        text = ""

ext = feature_extraction()
#print(corpus)
print('bow extraction')
bow_vectorizer, bow_features = pre.bow_extraction(corpus, ext)
print('tf-idf transform')
tfidf_trans, tfidf_features = pre.tfidf_extraction(corpus, ext, bow_vectorizer, bow_features)

print('tf-idf directly:')
pre.tfidf_extraction_directly(ext, corpus, bow_vectorizer)

print('new doc')
new_doc = ['loving this blue sky today']
pre.tfidf_new_doc_features(new_doc, ext, bow_vectorizer, tfidf_trans)

#generates matrix to apply the word2vec model.
corpus_w2v = []
for reg in classifier.filtered_list_lemma:
    for r in reg:
        corpus_w2v.append(r)

print('word2vec')
# build the word2vec model on our training corpus
model = ext.create_model_word2vec(corpus_w2v, size=500, window=100, min_count=30, sample=1e-3)

# get averaged word vectors for our training CORPUS
avg_word_vec_features = ext.averaged_word_vectorizer(corpus=corpus_w2v, model=model, num_features=500)
print (np.round(avg_word_vec_features, 3))
avg_word_vec_features_new_doc = ext.averaged_word_vectorizer(corpus=new_doc, model=model, num_features=500)
print (np.round(avg_word_vec_features_new_doc, 3))

print('tfidf_weighted_averaged:')
tfidf_vectorizer, tfidf_features = ext.tfidf_extractor(corpus)
#get tfidf weights and vocabulary from earlier results and compute result
corpus_tfidf = tfidf_features
vocab = tfidf_vectorizer.vocabulary_
wt_tfidf_word_vec_features = ext.tfidf_weighted_averaged_word_vectorizer(corpus=corpus_w2v, tfidf_vectors=corpus_tfidf
                                                                             ,tfidf_vocabulary=vocab, model=model,num_features=500)
print (np.round(wt_tfidf_word_vec_features, 3))

'''