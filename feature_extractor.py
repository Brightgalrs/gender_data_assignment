# extract features from list of text instances based on configuration set of features

import nltk
import numpy
import re
import csv
import gensim
import operator
import collections

source_text = []
stemmed_text = []
tokenized_text = []
pos_text = []
count_text = []
no_stop_text = []
no_stop_text2 = []

quote_text = []
no_stop_text3 = []


def preprocess():
	# first stem and lowercase words, then remove rare
	# lowercase
	global source_text
	source_text = [text.lower() for text in source_text]

	global count_text
	count_text = [len(text.split()) for text in source_text]

	# tokenize
	global tokenized_text
	tokenized_text = [nltk.word_tokenize(text) for text in source_text]

	# stem
	porter = nltk.PorterStemmer()
	global stemmed_text
	stemmed_text = [[porter.stem(t) for t in tokens] for tokens in tokenized_text]

	# remove rare
	vocab = nltk.FreqDist(w for line in stemmed_text for w in line)
	rarewords_list = set(vocab.hapaxes())
	stemmed_text = [['<RARE>' if w in rarewords_list else w for w in line] for line in stemmed_text]

	# pos
	global pos_text
	pos_text = [[a[1] for a in nltk.pos_tag(text)] for text in tokenized_text]

	# stop words
	global no_stop_text
	no_stop_text = [[word for word in text if word not in nltk.corpus.stopwords.words('english')] for text in stemmed_text]

	global no_stop_text2
	no_stop_text2 = [[word for word in text if word not in nltk.corpus.stopwords.words('english')] for text in tokenized_text]

	# pre-processing for quoted text
	global quote_text
	quote_text = [text.lower() for text in quote_text]

	tokenized_quotes = [nltk.word_tokenize(text) for text in quote_text]

	global no_stop_text3
	no_stop_text3 = [[word for word in text if word not in nltk.corpus.stopwords.words('english')] for text in tokenized_quotes]

# avg word vec of responses
def avg_word_vec():
	model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
	out = []
	for text in no_stop_text2:
		avg = numpy.zeros(300)
		cnt = 0
		for word in text:
			if (word in model.vocab):
				avg = numpy.add(avg, model[word])
				cnt += 1
		if cnt == 0:
			avg_vec = avg
		else:
			avg_vec = avg / cnt
		out.append(avg_vec.tolist())
	return numpy.asarray(out).T.tolist()


# avg word vec of quotes
def my_feature():
	model2 = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
	out2 = []
	for text2 in no_stop_text3:
		avg2 = numpy.zeros(300)
		cnt2 = 0
		for word2 in text2:
			if (word2 in model2.vocab):
				avg2 = numpy.add(avg2, model2[word2])
				cnt2 += 1
		if cnt2 == 0:
			avg_vec2 = avg2
		else:
			avg_vec2 = avg2 / cnt2
		out2.append(avg_vec2.tolist())
	return numpy.asarray(out2).T.tolist()


def bag_of_function_words():
	bow = []
	for sw in nltk.corpus.stopwords.words('english'):
		counts = [sum(1 for _ in re.finditer(r'\b%s\b' % sw, line)) for line in source_text]
		counts = [float(counts[i]) / count_text[i] for i in range(0, len(counts))]
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
		out.append([float(counts[i]) / count_text[i] for i in range(0, len(texts))])
	return out


def extract_features(text, quotes, conf):
	allf = False
	if len(conf)==0:
		allf = True

	global source_text
	source_text = text			# we'll use global variables to pass the data around

	global quote_text
	quote_text = quotes

	preprocess()

	features = []		# features will be list of lists, each component list will have the same length as the list of input text
	header = []

	# extract requested features: FILL IN HERE
	if 'word_vec' in conf or allf:
		features.extend(avg_word_vec())
	if 'my_feature' in conf or allf:
		features.extend(my_feature())
	'''if 'bag_of_function_words' in conf or allf:
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
		features.extend(bag_of_ngrams(1, no_stop_text, 100))'''

	features = numpy.asarray(features).T.tolist() # transpose list of lists so its dimensions are #instances x #features

	return features
