# label an argument response as sarcastic or nonsarcastic

from sklearn.model_selection import cross_val_score
from sklearn import svm
import feature_extractor as fe
import numpy
import re
import sys
import csv
import datetime

datafile = "data/sarcasm_v2.csv"
conffile = sys.argv[1]

ndata = -1 # for testing feature extraction: optional arg to control how much of data to use. won't work for testing classification because it just takes the first n -- all one class
if len(sys.argv) > 2:
	ndata = int(sys.argv[2])

def load_data():
	with open(datafile) as f:
		ls = list(csv.reader(f))[0:ndata]
		return [[b.decode('UTF-8') for b in a] for a in ls]

def load_conf_file():
	conf = set(line.strip() for line in open(conffile))
	return conf

def predict_sarcasm(X, Y):
	scores = cross_val_score(svm.SVC(), X, Y, scoring='accuracy', cv=10)
	return scores.mean()

if __name__ == "__main__":
	data = load_data()
	conf = load_conf_file()
	features = fe.extract_features([line[-1] for line in data if line[0]=="GEN"], [line[-2] for line in data if line[0]=="GEN"], conf)
	labels = [line[1] for line in data if line[0]=="GEN"]

	score = predict_sarcasm(features, labels)
	print(score)

	with open('log.csv', 'a') as f:
		f.write(",".join(('{:%Y-%m-%d,%H:%M:%S}'.format(datetime.datetime.now()), "|".join(conf), str(score))))
