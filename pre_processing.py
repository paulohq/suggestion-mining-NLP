# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd
import csv
import nltk
from pprint import pprint
import re
import string
import tkinter
#from nltk.corpus import wordnet # To get words in dictionary with their parts of speech
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from collections import Counter
from corretor_ortografico_norvig import *

from feature_extraction import *
from contractions import contractions_dict



from nltk.corpus import wordnet as wn
from contractions import contractions_dict
import re
import nltk
import string
from nltk.stem import WordNetLemmatizer
from pattern.en import tag


class pre_processing(object):
    def __init__(self):
        self.text = ''
        self.stopword_list = nltk.corpus.stopwords.words('english')
        self.wnl = WordNetLemmatizer()

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

        lemma_tokens = [wnl.lemmatize(token, self.get_pos(token)) for token in tokens]

        #lemma_tokens = [wnl.lemmatize(token, POS_tag(token)) if POS_tag(token) else token for token in tokens]
        return lemma_tokens

    #Speel checker the tokens.
    def correction(self, tokens):
        corretor = corretor_ortografico_norvig()
        filtered_tokens = [corretor.correction(token) for token in tokens]

        return filtered_tokens

    ###########################
    def tokenize_text1(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = [token.strip() for token in tokens]
        return tokens

    def expand_contractions1(self, text, contraction_mapping):
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
        flags = re.IGNORECASE | re.DOTALL)

        #def expand_match(contraction):
        #    match = contraction.group(0)
        #    first_char = match[0]
        #    expanded_contraction = contraction_mapping.get(match) \
        #        if contraction_mapping.get(match) \
        #        else contraction_mapping.get(match.lower())
        #    expanded_contraction = first_char + expanded_contraction[1:]
        #    return expanded_contraction

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

    # Annotate text tokens with POS tags
    def pos_tag_text1(self,text):

        # convert Penn treebank tag to wordnet tag
        def penn_to_wn_tags(self, pos_tag):
            if pos_tag.startswith('J'):
                return wn.ADJ
            elif pos_tag.startswith('V'):
                return wn.VERB
            elif pos_tag.startswith('N'):
                return wn.NOUN
            elif pos_tag.startswith('R'):
                return wn.ADV
            else:
                return None

        tagged_text = tag(text)
        tagged_lower_text = [(word.lower(), penn_to_wn_tags(self, pos_tag))
                             for word, pos_tag in tagged_text]
        return tagged_lower_text

    # lemmatize text based on POS tags
    def lemmatize_text1(self,text):
        pos_tagged_text = self.pos_tag_text1(text)
        lemmatized_tokens = [self.wnl.lemmatize(word, pos_tag) if pos_tag
                             else word
                             for word, pos_tag in pos_tagged_text]
        lemmatized_text = ' '.join(lemmatized_tokens)
        return lemmatized_text

    def remove_stopwords1(self, text):
        tokens = self.tokenize_text1(text)
        filtered_tokens = [token for token in tokens if token not in
                            self.stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def remove_special_characters1(self,text):
        tokens = self.tokenize_text1(text)
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
        filtered_tokens = filter(None, [pattern.sub('', token) for token in
                                         tokens])
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def normalize_corpus(self, corpus, tokenize=False):
        normalized_corpus = []
        for text in corpus:
            text = self.expand_contractions1(text, contractions_dict)
            text = self.lemmatize_text1(text)
            text = self.remove_special_characters1(text)
            text = self.remove_stopwords1(text)
            normalized_corpus.append(text)
            if tokenize:
                text = self.tokenize_text1(text)
                normalized_corpus.append(text)
        return normalized_corpus

    #def normalize_corpus(self, corpus, tokenize=False):
    #    normalized_corpus = []
    #    for text in corpus:
    #        text = self.expand_contractions1(text, contractions_dict)

    #        text = self.tokenize_text1(text)
    #        text = self.remove_special_characters1(text)
    #        text = self.lemmatizer(text)
    #        text = self.remove_stopwords1(text)
    #        normalized_corpus.append(text)

    #    return normalized_corpus
    #####################

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

    def tfidf_extraction_directly(self, ext, corpus, bow_vectorizer):
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


pre = pre_processing()
