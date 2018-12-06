#teste
import re, collections
from collections import Counter


def tokens(text):
    #Get all words from the corpus

    return re.findall('[a-z]+', text.lower())

f = open('/home/paulo/PycharmProjects/suggestion-mining/big.txt', 'r')
WORDS = tokens(f.read())
WORD_COUNTS = collections.Counter(WORDS)

print (WORD_COUNTS.most_common(10))

def edits0(word):
    #Return all strings that are zero edits away from the input word (i.e., the word itself).
    return {word}

def edits1(word):
    #Return all strings that are one edit away from the input word.
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    def splits(word):
        #Return a list of all possible (first, rest) pairs that the input word is made of.
        return [(word[:i], word[i:])
            for i in range(len(word)+1)]
        pairs = splits(word)
        deletes = [a + b[1:] for (a, b) in pairs if b]
        transposes = [a + b[1] + b[0] + b[2:] for (a, b) in pairs if len(b) > 1]
        replaces = [a+c+b[1:] for (a, b) in pairs for c in alphabet if b]
        inserts = [a+c+b for (a,b) in pairs for c in alphabet]

        return set(deletes + transposes + replaces + inserts)

def edits2(word):
    #Return all strings that are two edits away from the input word.
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

def known(words):
    #Return the subset of words that are actually in our WORD_COUNTS dictionary.
    return {w for w in words if w in WORD_COUNTS}

# input word
word = 'fianlly'
# zero edit distance from input word
edits0(word)
# returns null set since it is not a valid word
known(edits0(word))

# one edit distance from input word
edits1(word)
# get correct words from above set
known(edits1(word))

# two edit distances from input word
edits2(word)

# get correct words from above set
known(edits2(word))



def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('big.txt').read()))

def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N

def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

correction('speling')