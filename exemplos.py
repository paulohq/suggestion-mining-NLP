import nltk
import re
import string
from pprint import pprint
from nltk.corpus import wordnet
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer # lemmatizes word based on it's parts of speech
from nltk import pos_tag, word_tokenize
import collections
from collections import Counter
from nltk.corpus import wordnet # To get words in dictionary with their parts of speech
from corretor_ortografico_norvig import *

from contractions import contractions_dict
from feature_extraction import *


class exemplo(object):
    def __init__(self):
        self.corpus = ["i'm The brown brown fox wasn't that quick and he couldn't win the race.''",
                    "Hey that's a great deal! I just not bought a phone for $199 ",
                    "@@You'll (learn) a **lot** in the book. Python is an amazinngggg language !@@",
                    "My schooool is realllllyyy amaaazingggg"]

        self.list_spell_checker = []
        self.token_list = []
        self.filtered_list = []
        self.filtered_list_lowercase = []
        self.filtered_list_remove_stopwords = []
        self.filtered_list_remove_repeated_characters = []
        self.filtered_list_expand_contractions = []
        self.filtered_list_stemmer = []
        self.filtered_list_lemma = []

    def expand_contractions(self, text, contraction_mapping):
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
        flags = re.IGNORECASE | re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            #expanded_contraction = contraction_mapping.get(match) \
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

    def tokenize_text(self,text):
        sentences = nltk.sent_tokenize(text)
        word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
        return word_tokens

    def remove_characters_after_tokenization(self,tokens):
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
        #for token in tokens:
        #    filtered_tokens = list(filter(None, [pattern.sub('', token) ]))
        filtered_tokens = list(filter(None, [pattern.sub('', token) for token in tokens]))
        #print (filtered_tokens)
        return filtered_tokens

    def lower_case(self, token_list):
        lower_token_list = [token.lower() for token in token_list]
        return lower_token_list

    def remove_stopwords(self, tokens):
        stopword_list = nltk.corpus.stopwords.words('english')
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        return filtered_tokens

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

    def lancaster_stemmer(self, tokens):
        ls = LancasterStemmer()
        filtered_tokens = [ls.stem(token) for token in tokens]
        return filtered_tokens


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

    def bow_extraction(self, corpus, ext):

        bow_vectorizer, bow_features = ext.bow_extractor(corpus)
        features = bow_features.todense()
        feature_names = bow_vectorizer.get_feature_names()
        df = ext.display_features(features, feature_names)
        return bow_vectorizer, bow_features

        #print(df)

    #TF-IDF extraction
    def tfidf_extraction(self, corpus, ext, bow_vectorizer, bow_features):
        #ext = feature_extraction()
        #bow_vectorizer, bow_features = ext.bow_extractor(corpus)

        feature_names = bow_vectorizer.get_feature_names()
        tfidf_trans, tfidf_features = ext.tfidf_transformer(bow_features)
        features = np.round(tfidf_features.todense(), 2)

        #bow_vectorizer, features, feature_names = ext.tfid_extractor(corpus)
        df = ext.display_features(features, feature_names)
        return tfidf_trans, tfidf_features

    def tfidf_extraction_directly(self, ext, corpus):
        tfidf_vectorizer, tdidf_features = ext.tfidf_extractor(corpus)
        feature_names = bow_vectorizer.get_feature_names()
        ext.display_features(np.round(tdidf_features.todense(), 2), feature_names)


    def create_model_word2vec(self, tokenized_corpus):
        ext = feature_extraction()
        ext.create_model_word2vec(tokenized_corpus, 10, 10, 2, 1e-3)


exem = exemplo()
print("Original corpus:")
print(exem.corpus)
exem.filtered_list_expand_contractions = [exem.expand_contractions(text, contractions_dict) for text in exem.corpus]
print("Expand contractions:")
print(exem.filtered_list_expand_contractions)




exem.token_list = [exem.tokenize_text(text) for text in exem.filtered_list_expand_contractions]
print("Token List:")
print(exem.token_list)

#filtered_list_1 = [[remove_characters_after_tokenization(tokens)
#                                for tokens in sentence_tokens]
#                                    for sentence_tokens in token_list]

#Loop to remove special characters from the list token_list.
for sentence_tokens in exem.token_list:
    for tokens in sentence_tokens:
        exem.filtered_list.append(list(filter(None, [exem.remove_characters_after_tokenization(tokens)])))
#Print list after remove special characters.
#print(filtered_list)

#Loop to convert words in filtered_list to lowercase.
for sentence_tokens in exem.filtered_list:
    for tokens in sentence_tokens:
        exem.filtered_list_lowercase.append(list(filter(None, [exem.lower_case(tokens)])))
#Print list after convert to lowercase.
#print(filtered_list_lowercase)

#Loop to remove the stopwords from the list filtered_list_lowercase
for sentence_tokens in exem.filtered_list_lowercase:
    for tokens in sentence_tokens:
        exem.filtered_list_remove_stopwords.append(list(filter(None, [exem.remove_stopwords(tokens)])))

#Print list after remove stopwords
#print (filtered_list_remove_stopwords)

#Loop to remove the repeated characters from the list filtered_list_remove_stopwords.
for sentence_tokens in exem.filtered_list_remove_stopwords:
    for tokens in sentence_tokens:
        exem.filtered_list_remove_repeated_characters.append(list(filter(None, [exem.remove_repeated_characters(tokens)])))
#Print list after remove repeated characters.
print("Stopwords removal:")
print(exem.filtered_list_remove_repeated_characters)


#Loop to spell checker.
#for sentence_tokens in exem.filtered_list_remove_repeated_characters:
#    for tokens in sentence_tokens:
#        exem.list_spell_checker.append((list(filter(None, [exem.correction(tokens)]))))
#print('Speel checker:')
#print(exem.list_spell_checker)

#Loop to stemming.
for sentence_tokens in exem.filtered_list_remove_repeated_characters:
    for tokens in sentence_tokens:
        exem.filtered_list_stemmer.append(list(filter(None, [exem.lancaster_stemmer(tokens)])))
print('Stem:')
#Print list after stemming.
print(exem.filtered_list_stemmer)

for sentence_tokens in exem.filtered_list_remove_repeated_characters:
    for tokens in sentence_tokens:
        exem.filtered_list_lemma.append(list(filter(None, [exem.lemmatizer(tokens)])))

print('Lemma:')
print(exem.filtered_list_lemma)

#generate vector with the sentences to apply tf-idf model.
text = ""
corpus = []
for reg in exem.filtered_list_lemma:
    for r in reg:
        for i in r:
            text = text + ' ' + i
        corpus.append(text)
        text = ""

ext = feature_extraction()
#print(corpus)
print('bow extraction')
bow_vectorizer, bow_features = exem.bow_extraction(corpus, ext)
print('tf-idf transform')
tfidf_trans, tfidf_features = exem.tfidf_extraction(corpus, ext, bow_vectorizer, bow_features)

print('tf-idf directly:')
exem.tfidf_extraction_directly(ext, corpus)


#generates matrix to apply the word2vec model.
corpus_w2v = []
for reg in exem.filtered_list_lemma:
    for r in reg:
        corpus_w2v.append(r)
#print('word2vec model')
#exem.create_model_word2vec(corpus1)
print('word2vec')
# build the word2vec model on our training corpus
model = ext.create_model_word2vec(corpus_w2v, size=10, window=10, min_count=2, sample=1e-3)

# get averaged word vectors for our training CORPUS
avg_word_vec_features = extract.averaged_word_vectorizer(corpus=corpus_w2v, model=model, num_features=10)
print (np.round(avg_word_vec_features, 3))
avg_word_vec_features_new_doc = extract.averaged_word_vectorizer(corpus=new_doc, model=model, num_features=10)
print (np.round(avg_word_vec_features_new_doc, 3))

print('tfidf_weighted_averaged:')
tfidf_vectorizer, tfidf_features = ext.tfidf_extractor(corpus)
#get tfidf weights and vocabulary from earlier results and compute result
corpus_tfidf = tfidf_features
vocab = tfidf_vectorizer.vocabulary_
wt_tfidf_word_vec_features = ext.tfidf_weighted_averaged_word_vectorizer(corpus=corpus_w2v, tfidf_vectors=corpus_tfidf
                                                                             ,tfidf_vocabulary=vocab, model=model,num_features=10)
print (np.round(wt_tfidf_word_vec_features, 3))
