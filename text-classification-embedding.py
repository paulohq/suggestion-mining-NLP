# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
from feature_extraction import *
from pre_processing import *
from read_csv import *
from contractions import contractions_dict

class text_classification(object):
    def __init__(self):
        #train = file_reader = csv.reader(open(data_path, "rt", errors="ignore", encoding="utf-8"), delimiter=',')
        #self.data_path = "/home/paulo/PycharmProjects/suggestion-mining/training-full-v13-bkp.csv"
        #train = pd.read_csv(data_path, header=0, sep=";", quotechar='"',quoting=3, encoding='utf-8')

        self.filtered_list_expand_contractions = []
        self.sent_list = []
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
pre = pre_processing()
classifier.sent_list = read.read_csv()
#sent_tokenize(sent_list)
#word_tokenize(sent_list)

#print(classifier.sent_list[0][1])
#for text in classifier.sent_list:
#    print(text[1])

classifier.filtered_list_expand_contractions = [pre.expand_contractions(text[1], contractions_dict) for text in classifier.sent_list]
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

