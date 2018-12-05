import nltk
import re
import string
from pprint import pprint
from nltk.corpus import wordnet
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer # lemmatizes word based on it's parts of speech
from nltk import pos_tag, word_tokenize

from collections import Counter

from nltk.corpus import wordnet # To get words in dictionary with their parts of speech



corpus = ["The brown fox wasn't that quick and he couldn't win the race.''",
"Hey that's a great deal! I just not bought a phone for $199 ",
"@@You'll (learn) a **lot** in the book. Python is an amazinngggg language !@@",
"My schooool is realllllyyy amaaazingggg"]

filtered_list = []
filtered_list_lowercase = []
filtered_list_remove_stopwords = []
filtered_list_remove_repeated_characters = []
filtered_list_stemmer = []
filtered_list_lemma = []

def tokenize_text(text):
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return word_tokens

def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    #for token in tokens:
    #    filtered_tokens = list(filter(None, [pattern.sub('', token) ]))
    filtered_tokens = list(filter(None, [pattern.sub('', token) for token in tokens]))
    #print (filtered_tokens)
    return filtered_tokens

def lower_case(token_list):
    lower_token_list = [token.lower() for token in token_list]
    return lower_token_list

def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens

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

def lancaster_stemmer(tokens):
    ls = LancasterStemmer()
    filtered_tokens = [ls.stem(token) for token in tokens]
    return filtered_tokens


def get_pos(word):
    w_synsets = wordnet.synsets(word)

    pos_counts = Counter()
    pos_counts["n"] = len([item for item in w_synsets if item.pos() == "n"])
    pos_counts["v"] = len([item for item in w_synsets if item.pos() == "v"])
    pos_counts["a"] = len([item for item in w_synsets if item.pos() == "a"])
    pos_counts["r"] = len([item for item in w_synsets if item.pos() == "r"])

    most_common_pos_list = pos_counts.most_common(3)
    return most_common_pos_list[0][0]  # first indexer for getting the top POS from list, second indexer for getting POS from tuple( POS: count )


#Convert the tokens to your radical (root word) using tokens POS-tag before lemmatizing.
def lemmatizer(tokens):
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

    lemma_tokens = [wnl.lemmatize(token, get_pos(token)) for token in tokens]

    #lemma_tokens = [wnl.lemmatize(token, POS_tag(token)) if POS_tag(token) else token for token in tokens]
    return lemma_tokens

token_list = [tokenize_text(text) for text in corpus]

pprint(token_list)

#filtered_list_1 = [[remove_characters_after_tokenization(tokens)
#                                for tokens in sentence_tokens]
#                                    for sentence_tokens in token_list]

#Loop to remove special characters from the list token_list.
for sentence_tokens in token_list:
    for tokens in sentence_tokens:
        filtered_list.append(list(filter(None, [remove_characters_after_tokenization(tokens)])))
#Print list after remove special characters.
#print(filtered_list)

#Loop to convert words in filtered_list to lowercase.
for sentence_tokens in filtered_list:
    for tokens in sentence_tokens:
        filtered_list_lowercase.append(list(filter(None, [lower_case(tokens)])))
#Print list after convert to lowercase.
#print(filtered_list_lowercase)

#Loop to remove the stopwords from the list filtered_list_lowercase
for sentence_tokens in filtered_list_lowercase:
    for tokens in sentence_tokens:
        filtered_list_remove_stopwords.append(list(filter(None, [remove_stopwords(tokens)])))

#Print list after remove stopwords
#print (filtered_list_remove_stopwords)

#Loop to remove the repeated characters from the list filtered_list_remove_stopwords.
for sentence_tokens in filtered_list_remove_stopwords:
    for tokens in sentence_tokens:
        filtered_list_remove_repeated_characters.append(list(filter(None, [remove_repeated_characters(tokens)])))
#Print list after remove repeated characters.
print(filtered_list_remove_repeated_characters)

#Loop to stemming.
for sentence_tokens in filtered_list_remove_repeated_characters:
    for tokens in sentence_tokens:
        filtered_list_stemmer.append(list(filter(None, [lancaster_stemmer(tokens)])))
#Print list after stemming.
print(filtered_list_stemmer)

for sentence_tokens in filtered_list_remove_repeated_characters:
    for tokens in sentence_tokens:
        filtered_list_lemma.append(list(filter(None, [lemmatizer(tokens)])))

print(filtered_list_lemma)