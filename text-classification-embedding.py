# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd
import csv
import nltk
from pprint import pprint
import re
import string

#train = file_reader = csv.reader(open(data_path, "rt", errors="ignore", encoding="utf-8"), delimiter=',')
data_path = "/home/paulo/PycharmProjects/suggestion-mining/training-full-v13-bkp.csv"
#train = pd.read_csv(data_path, header=0, sep=";", quotechar='"',quoting=3, encoding='utf-8')

# This reads CSV a given CSV and stores the data in a list
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

def word_tokenize(sentence):
    default_w = nltk.word_tokenize
    word_tokens = [default_w(sentence)]
    return word_tokens

def remove_characters_after_tokenization(token_list):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    for tokens in token_list:
        for token in tokens:
            filtered_tokens = [filter(None, [pattern.sub('', token)])]
    #filtered_tokens = filter(None, [pattern.sub('', token) for token in token_list])
    return filtered_tokens

sent_list = read_csv(data_path)
#sent_tokenize(sent_list)
#word_tokenize(sent_list)
token_list = [word_tokenize(sent[1]) for sent in sent_list]
pprint(token_list)

filtered_list_1 = remove_characters_after_tokenization(token_list)
#filtered_list_1 = [filter(None,[remove_characters_after_tokenization(tokens)
#                                for tokens in sentence_tokens])
#                            for sentence_tokens in token_list]
print(filtered_list_1)