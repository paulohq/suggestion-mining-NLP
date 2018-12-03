# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd
import csv
import nltk
from pprint import pprint
import re
import string
import tkinter
from nltk.corpus import wordnet
from nltk.stem import LancasterStemmer

#train = file_reader = csv.reader(open(data_path, "rt", errors="ignore", encoding="utf-8"), delimiter=',')
data_path = "/home/paulo/PycharmProjects/suggestion-mining/training-full-v13-bkp.csv"
#train = pd.read_csv(data_path, header=0, sep=";", quotechar='"',quoting=3, encoding='utf-8')

filtered_list = []
filtered_list_lowercase = []
filtered_list_remove_stopwords = []
filtered_list_remove_repeated_characters = []
filtered_list_stemmer = []

# Reads a given CSV and stores the data in a list
def read_csv(data_path):
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
def sent_tokenize(sent_list):
    default_st = nltk.sent_tokenize
    for sent in sent_list:
        sentence = default_st(text=sent[1])
        print(sentence)

def word_tokenize1(sent_list):
    default_w = nltk.word_tokenize
    for sent in sent_list:
        token_list = [default_w(sent[1])]
        print(token_list[0])

#Tokenize the sentence into words.
def word_tokenize(sentence):
    default_w = nltk.word_tokenize
    word_tokens = [default_w(sentence)]
    return word_tokens

# Remove special characters after tokenization
def remove_characters_after_tokenization(token_list):
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
def lower_case(token_list):
    lower_token_list = [token.lower() for token in token_list]
    return lower_token_list

#Remove the stopwords from the token list.
def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens

#Remove the repeated characters from the token list.
def remove_repeated_characters(tokens):
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
def lancaster_stemmer(tokens):
    ls = LancasterStemmer()
    filtered_tokens = [ls.stem(token) for token in tokens]
    return filtered_tokens

sent_list = read_csv(data_path)
#sent_tokenize(sent_list)
#word_tokenize(sent_list)
token_list = [word_tokenize(sent[1]) for sent in sent_list]
#Print list after execute word tokenize.
#pprint(token_list)

#Loop to remove special characters from the list token_list.
for sentence_tokens in token_list:
    for tokens in sentence_tokens:
        filtered_list.append(list(filter(None, [remove_characters_after_tokenization(tokens)])))
#Print list after remove special characters.
print(filtered_list)

#Loop to convert words in filtered_list to lowercase.
for sentence_tokens in filtered_list:
    for tokens in sentence_tokens:
        filtered_list_lowercase.append(list(filter(None, [lower_case(tokens)])))
#Print list after convert to lowercase.
print(filtered_list_lowercase)

#Loop to remove the stopwords from the list filtered_list_lowercase
for sentence_tokens in filtered_list_lowercase:
    for tokens in sentence_tokens:
        filtered_list_remove_stopwords.append(list(filter(None, [remove_stopwords(tokens)])))

#Print list after remove stopwords
print (filtered_list_remove_stopwords)

#Loop to remove the repeated characters from the list filtered_list_remove_stopwords.
for sentence_tokens in filtered_list_remove_stopwords:
    for tokens in sentence_tokens:
        filtered_list_remove_repeated_characters.append(list(filter(None, [remove_repeated_characters(tokens)])))
#Print list after remove repeated characters.
for sentence_tokens in filtered_list_remove_repeated_characters:
    for tokens in sentence_tokens:
      print(tokens)

#Loop to stemming.
for sentence_tokens in filtered_list_remove_repeated_characters:
    for tokens in sentence_tokens:
        filtered_list_stemmer.append(list(filter(None, [lancaster_stemmer(tokens)])))
#Print list after stemming.
print(filtered_list_stemmer)