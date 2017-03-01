# extract features from list of text instances based on configuration set of features

import nltk
import re
import gensim
import operator
import collections
import numpy

source_text = []
stemmed_text = []
tokenized_text = []
pos_text = []
count_text = []



def preprocess():
    # first stem and lowercase words, then remove rare
    # lowercase
    global source_text
    source_text = [text.lower() for text in source_text]

    # tokenize
    tokenized_text = [nltk.word_tokenize(text) for text in source_text]

    # stem
    porter = nltk.PorterStemmer()
    global stemmed_text
#    stemmed_text = [[porter.stem(t) for t in tokens] for tokens in tokenized_text]
    for tokens in tokenized_text:  # iterating instead of list comprehension to allow exception handling
        stemmed_line = []
        for t in tokens:
            try:
                stemmed_line.extend(porter.stem(t))
            except IndexError:
                stemmed_line.extend('')
        stemmed_text.append(stemmed_line)

    # remove rare
#    vocab = nltk.FreqDist(w for w in line for line in stemmed_text)
    vocab = nltk.FreqDist(w for line in stemmed_text for w in line)
    rarewords_list = vocab.hapaxes()
    rarewords_regex = re.compile(r'\b(%s)\b' % '|'.join(rarewords_list))
    stemmed_text = [[rarewords_regex.sub('<RARE>', w) for w in line] for line in stemmed_text]
    # note that source_text will be lowercased, but only stemmed_text will have rare words removed


def bag_of_function_words():
    bow = []
    for sw in nltk.corpus.stopwords.words('english'):
        counts = [sum(1 for _ in re.finditer(r'\b%s\b' % sw, line)) for line in source_text]
        bow.append(counts)
    return bow


# FILL IN OTHER FEATURE EXTRACTORS
def extract_features(text, conf):
    all = False
    if len(conf) == 0:
        all = True

    global source_text
    source_text = text            # we'll use global variables to pass the data around
    preprocess()

    features = []        # features will be list of lists, each component list will have the same length as the list of input text

    # extract requested features: FILL IN HERE
    if 'bag_of_function_words' in conf or all:
        features.extend(bag_of_function_words())

    features = numpy.asarray(features).T.tolist()  # transpose list of lists sow its dimensions are #instances x #features

    return features
