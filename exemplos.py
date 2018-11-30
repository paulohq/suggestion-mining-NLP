import nltk
import re
import string
from pprint import pprint


corpus = ["The brown fox wasn't that quick and he couldn't win the race",
"Hey that's a great deal! I just bought a phone for $199",
"@@You'll (learn) a **lot** in the book. Python is an amazing language !@@"]


def tokenize_text(text):
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return word_tokens

def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    for token in tokens:
        filtered_tokens = list(filter(None, [pattern.sub('', token) ]))
    #filtered_tokens = list(filter(None, [pattern.sub('', token) for token in tokens]))
    #print (filtered_tokens)
    return filtered_tokens

token_list = [tokenize_text(text) for text in corpus]

pprint(token_list)

#filtered_list_1 = [[remove_characters_after_tokenization(tokens)
#                                for tokens in sentence_tokens]
#                                    for sentence_tokens in token_list]

filtered_list_1 = list([filter(None,[remove_characters_after_tokenization(tokens)
                                for tokens in sentence_tokens])
                                    for sentence_tokens in token_list])
print (filtered_list_1)