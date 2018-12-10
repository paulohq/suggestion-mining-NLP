# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd
import csv
import nltk
from pprint import pprint
import re
import string
import tkinter
from nltk.corpus import wordnet # To get words in dictionary with their parts of speech
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from collections import Counter
from corretor_ortografico_norvig import *
from contractions import contractions_dict
from feature_extraction import *

class text_classification(object):
    def __init__(self):
        #train = file_reader = csv.reader(open(data_path, "rt", errors="ignore", encoding="utf-8"), delimiter=',')
        self.data_path = "/home/paulo/PycharmProjects/suggestion-mining/training-full-v13-bkp.csv"
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

    # Reads a given CSV and stores the data in a list
    def read_csv(self, data_path):
        file_reader = csv.reader(open(data_path, "rt", errors="ignore", encoding="utf-8"), delimiter=',')
        sent_list = []
        #print(file_reader.shape)
        #print(file_reader.columns.values)
        for row in file_reader:
            id = row[0]
            suggest = row[1]
            sent = row[2]
            sent_list.append((id, suggest, sent))
        return sent_list

    #Tokenize the text into sentences
    def sent_tokenize(self, sent_list):
        default_st = nltk.sent_tokenize
        for sent in sent_list:
            sentence = default_st(text=sent[1])
            print(sentence)

    def word_tokenize1(self, sent_list):
        default_w = nltk.word_tokenize
        for sent in sent_list:
            token_list = [default_w(sent[1])]
            print(token_list[0])

    #Expand the contractions in text.
    def expand_contractions(self, text, contraction_mapping):
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
        flags = re.IGNORECASE | re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = ""
            if contraction_mapping.get(match):
                expanded_contraction = contraction_mapping.get(match)
            elif contraction_mapping.get(match.lower()):
                expanded_contraction = contraction_mapping.get(match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text


    #Tokenize the sentence into words.
    def word_tokenize(self, sentence):
        default_w = nltk.word_tokenize
        word_tokens = [default_w(sentence)]
        return word_tokens

    # Remove special characters after tokenization
    def remove_characters_after_tokenization(self, token_list):
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
        #for token in token_list:
        #    filtered_tokens = list(filter(None, [pattern.sub('', token) ]))

    #    for tokens in token_list:
    #        for token in tokens:
    #            filtered_tokens = list([filter(None, [pattern.sub('', token)])])
        filtered_tokens = list(filter(None, [pattern.sub('', token) for token in token_list]))
        #print(filtered_tokens)

        return filtered_tokens

    #Convert tokens to lowercase.
    def lower_case(self, token_list):
        lower_token_list = [token.lower() for token in token_list]
        return lower_token_list

    #Remove the stopwords from the token list.
    def remove_stopwords(self, tokens):
        stopword_list = nltk.corpus.stopwords.words('english')
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        return filtered_tokens

    #Remove the repeated characters from the token list.
    #Identify repeated characters in a word using a regex pattern and then use a substitution to remove the characters one by one.
    def remove_repeated_characters(self, tokens):
        repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
        match_substitution = r'\1\2\3'
        def replace(old_word):
            if wordnet.synsets(old_word):
                return old_word

            new_word = repeat_pattern.sub(match_substitution, old_word)
            return replace(new_word) if new_word != old_word else new_word

        correct_tokens = [replace(word) for word in tokens]
        return correct_tokens

    #Stemming the tokens.
    def lancaster_stemmer(self, tokens):
        ls = LancasterStemmer()
        filtered_tokens = [ls.stem(token) for token in tokens]
        return filtered_tokens

    #POS-tag token
    def get_pos(self, word):
        w_synsets = wordnet.synsets(word)

        pos_counts = Counter()
        pos_counts["n"] = len([item for item in w_synsets if item.pos() == "n"])
        pos_counts["v"] = len([item for item in w_synsets if item.pos() == "v"])
        pos_counts["a"] = len([item for item in w_synsets if item.pos() == "a"])
        pos_counts["r"] = len([item for item in w_synsets if item.pos() == "r"])

        most_common_pos_list = pos_counts.most_common(3)
        return most_common_pos_list[0][0]  # first indexer for getting the top POS from list, second indexer for getting POS from tuple( POS: count )


    #Convert the tokens to your radical (root word) using tokens POS-tag before lemmatizing.
    def lemmatizer(self, tokens):
        wnl = WordNetLemmatizer()

        def POS_tag(token):
            tag = pos_tag(token)
            wntag = tag[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
            return wntag

        #for token, tag in pos_tag(tokens):
        #    wntag = tag[0].lower()
        #    wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
            #lemma = wnl.lemmatize(token, wntag) if wntag else token

        lemma_tokens = [wnl.lemmatize(token, self.get_pos(token)) for token in tokens]

        #lemma_tokens = [wnl.lemmatize(token, POS_tag(token)) if POS_tag(token) else token for token in tokens]
        return lemma_tokens

    #Speel checker the tokens.
    def correction(self, tokens):
        corretor = corretor_ortografico_norvig()
        filtered_tokens = [corretor.correction(token) for token in tokens]

        return filtered_tokens

    # Bag of words extraction.
    def bow_extraction(self, corpus, ext):

        bow_vectorizer, bow_features = ext.bow_extractor(corpus)
        features = bow_features.todense()
        feature_names = bow_vectorizer.get_feature_names()
        df = ext.display_features(features, feature_names)
        return bow_vectorizer, bow_features

    #TF-IDF extraction
    def tfidf_extraction(self, corpus, ext, bow_vectorizer, bow_features):
        feature_names = bow_vectorizer.get_feature_names()
        tfidf_trans, tfidf_features = ext.tfidf_transformer(bow_features)
        features = np.round(tfidf_features.todense(), 2)

        df = ext.display_features(features, feature_names)
        return tfidf_trans, tfidf_features

    def tfidf_extraction_directly(self, ext, corpus):
        tfidf_vectorizer, tdidf_features = ext.tfidf_extractor(corpus)
        feature_names = bow_vectorizer.get_feature_names()
        ext.display_features(np.round(tdidf_features.todense(), 2), feature_names)

    def tfidf_new_doc_features(self, new_doc, ext, bow_vectorizer, tfidf_trans):
        nd_features, feature_names = ext.tfidf_new_doc_features(new_doc, bow_vectorizer, tfidf_trans)
        df = ext.display_features(nd_features, feature_names)

    #Method to create a word2vec model.
    def create_model_word2vec(self, tokenized_corpus):
        ext = feature_extraction()
        model = ext.create_model_word2vec(tokenized_corpus, size=10, window=10, min_count=2, sample=1e-3)

        # get averaged word vectors for our training CORPUS
        avg_word_vec_features = extract.averaged_word_vectorizer(corpus=tokenized_corpus, model=model, num_features=10)
        print(np.round(avg_word_vec_features, 3))
        return model, avg_word_vec_features


classifier = text_classification()
classifier.sent_list = classifier.read_csv(classifier.data_path)
#sent_tokenize(sent_list)
#word_tokenize(sent_list)

#print(classifier.sent_list[0][1])
#for text in classifier.sent_list:
#    print(text[1])

classifier.filtered_list_expand_contractions = [classifier.expand_contractions(text[1], contractions_dict) for text in classifier.sent_list]
print("Expand contractions:")
print(classifier.filtered_list_expand_contractions)

classifier.token_list = [classifier.word_tokenize(sent) for sent in classifier.filtered_list_expand_contractions]
#Print list after execute word tokenize.
#pprint(token_list)

#Loop to remove special characters from the list token_list.
for sentence_tokens in classifier.token_list:
    for tokens in sentence_tokens:
        classifier.filtered_list.append(list(filter(None, [classifier.remove_characters_after_tokenization(tokens)])))
#Print list after remove special characters.
#print(filtered_list)

#Loop to convert words in filtered_list to lowercase.
for sentence_tokens in classifier.filtered_list:
    for tokens in sentence_tokens:
        classifier.filtered_list_lowercase.append(list(filter(None, [classifier.lower_case(tokens)])))
#Print list after convert to lowercase.
#print(filtered_list_lowercase)

#Loop to remove the stopwords from the list filtered_list_lowercase
for sentence_tokens in classifier.filtered_list_lowercase:
    for tokens in sentence_tokens:
        classifier.filtered_list_remove_stopwords.append(list(filter(None, [classifier.remove_stopwords(tokens)])))

#Print list after remove stopwords
#print (filtered_list_remove_stopwords)

#Loop to remove the repeated characters from the list filtered_list_remove_stopwords.
for sentence_tokens in classifier.filtered_list_remove_stopwords:
    for tokens in sentence_tokens:
        classifier.filtered_list_remove_repeated_characters.append(list(filter(None, [classifier.remove_repeated_characters(tokens)])))
#Print list after remove repeated characters.
#for sentence_tokens in classifier.filtered_list_remove_repeated_characters:
#    for tokens in sentence_tokens:
#      print(tokens)

# Loop to spell checker.
#for sentence_tokens in classifier.filtered_list_remove_repeated_characters:
#    for tokens in sentence_tokens:
#        classifier.list_spell_checker.append((list(filter(None, [classifier.correction(tokens)]))))
#print('Speel checker:')
#print(classifier.list_spell_checker)

#Loop to stemming.
#for sentence_tokens in classifier.filtered_list_remove_repeated_characters:
#    for tokens in sentence_tokens:
#        classifier.filtered_list_stemmer.append(list(filter(None, [classifier.lancaster_stemmer(tokens)])))
#Print list after stemming.
#print(filtered_list_stemmer)


#Loop to lemmatize
for sentence_tokens in classifier.filtered_list_remove_repeated_characters:
    for tokens in sentence_tokens:
        classifier.filtered_list_lemma.append(list(filter(None, [classifier.lemmatizer(tokens)])))
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
bow_vectorizer, bow_features = classifier.bow_extraction(corpus, ext)
print('tf-idf transform')
tfidf_trans, tfidf_features = classifier.tfidf_extraction(corpus, ext, bow_vectorizer, bow_features)

print('tf-idf directly:')
classifier.tfidf_extraction_directly(ext, corpus)

print('new doc')
new_doc = ['loving this blue sky today']
classifier.tfidf_new_doc_features(new_doc, ext, bow_vectorizer, tfidf_trans)

#generates matrix to apply the word2vec model.
corpus_w2v = []
for reg in classifier.filtered_list_lemma:
    for r in reg:
        corpus_w2v.append(r)

print('word2vec')
# build the word2vec model on our training corpus
model = ext.create_model_word2vec(corpus_w2v, size=10, window=10, min_count=2, sample=1e-3)

# get averaged word vectors for our training CORPUS
avg_word_vec_features = ext.averaged_word_vectorizer(corpus=corpus_w2v, model=model, num_features=10)
print (np.round(avg_word_vec_features, 3))
avg_word_vec_features_new_doc = ext.averaged_word_vectorizer(corpus=new_doc, model=model, num_features=10)
print (np.round(avg_word_vec_features_new_doc, 3))

print('tfidf_weighted_averaged:')
tfidf_vectorizer, tfidf_features = ext.tfidf_extractor(corpus)
#get tfidf weights and vocabulary from earlier results and compute result
corpus_tfidf = tfidf_features
vocab = tfidf_vectorizer.vocabulary_
wt_tfidf_word_vec_features = ext.tfidf_weighted_averaged_word_vectorizer(corpus=corpus_w2v, tfidf_vectors=corpus_tfidf
                                                                             ,tfidf_vocabulary=vocab, model=model,num_features=10)
print (np.round(wt_tfidf_word_vec_features, 3))

