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

    global count_text
    count_text = [len(text.split()) for text in source_text]

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

    # pos
    global pos_text
    pos_text = [[a[1] for a in nltk.pos_tag(text)] for text in tokenized_text]

    # stop words
    global no_stop_text
    no_stop_text = [[word for word in text if word not in nltk.corpus.stopwords.words('english')] for text in stemmed_text]


def bag_of_function_words():
    bow = []
    for sw in nltk.corpus.stopwords.words('english'):
        counts = [sum(1 for _ in re.finditer(r'\b%s\b' % sw, line)) for line in source_text]
        counts = [counts[i] / count_text[i] for i in range(0, len(counts))]
        bow.append(counts)
    return bow


def bag_of_ngrams(n, texts, l):
    ngramss = [list(nltk.ngrams(text, n)) for text in texts]
    ngram_counts = []
    ngram_sum = collections.defaultdict(lambda: 0)

    for ngrams in ngramss:
        ngram_counts.append(collections.defaultdict(lambda: 0))
        for ngram in ngrams:
            ngram_counts[-1][ngram] += 1
            ngram_sum[ngram] += 1

    ngram_sort = sorted(ngram_sum.items(), key=operator.itemgetter(1), reverse=True)

    out = []
    for ngram, _ in ngram_sort[:l]:
        counts = [ngram_count[ngram] for ngram_count in ngram_counts]
        out.append([counts[i] / count_text[i] for i in range(0, len(texts))])
    return out


# topic modelling with latent semantic indexing
# https://radimrehurek.com/gensim/models/lsimodel.html
def topic_model_scores():
    d = gensim.corpora.Dictionary(stemmed_text)
    corpus = [d.doc2bow(text) for text in stemmed_text]

    tfidf = gensim.models.TfidfModel(corpus)

    corpus_tfidf = tfidf[corpus]

    lsi_model = gensim.models.lsimodel.LsiModel(corpus, num_topics=20, id2word=d)
    corpus_lsi = lsi_model[corpus_tfidf]

    return [[scores[i][1] for scores in corpus_lsi] for i in range(0, 20)]


def characters_per_word():
    return [[(len(source_text[i]) - count_text[i] + 1) / count_text[i] for i in range(0, len(count_text))]]


def unique_words_ratio():
    return [[len(set(text)) / len(text) for text in stemmed_text]]


def words_per_sentence():
    return [[len(text) / text.count('.') for text in pos_text]]


def extract_features(text, conf):
    allf = False
    if len(conf) == 0:
        allf = True

    global source_text
    source_text = text  # we'll use global variables to pass the data around
    preprocess()

    features = []  # features will be list of lists, each component list will have the same length as the list of input text

    # extract requested features
    if 'bag_of_function_words' in conf or allf:
        features.extend(bag_of_function_words())
    if 'bag_of_pos_trigrams' in conf or allf:
        features.extend(bag_of_ngrams(3, pos_text, 500))
    if 'bag_of_pos_bigrams' in conf or allf:
        features.extend(bag_of_ngrams(2, pos_text, 100))
    if 'bag_of_pos_unigrams' in conf or allf:
        features.extend(bag_of_ngrams(1, pos_text, 999))
    if 'bag_of_trigrams' in conf or allf:
        features.extend(bag_of_ngrams(3, stemmed_text, 500))
    if 'bag_of_bigrams' in conf or allf:
        features.extend(bag_of_ngrams(2, stemmed_text, 100))
    if 'bag_of_unigrams' in conf or allf:
        features.extend(bag_of_ngrams(1, no_stop_text, 100))
    if 'topic_model_scores' in conf or allf:
        features.extend(topic_model_scores())
    if 'characters_per_word' in conf or allf:
        features.extend(characters_per_word())
    if 'unique_words_ratio' in conf or allf:
        features.extend(unique_words_ratio())
    if 'words_per_sentence' in conf or allf:
        features.extend(words_per_sentence())

    features = numpy.asarray(features).T.tolist()  # transpose list of lists so its dimensions are #instances x #features

    return features
