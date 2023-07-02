#!/usr/bin/env python3

import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import argparse
from joblib import dump



def fix_format(token):
	if 'BoS' in token:
		token = token.replace('BoS','formPrev=BoS	suf3Prev=BoS')
	if 'EoS' in token:
		token = token.replace('EoS','formNext=EoS	suf3Next=EoS')

	return token


def load_data(data):
	features = []
	labels = []
	for token in data:
		token = token.strip()
		token = fix_format(token).split('\t')
		#token = token.split('\t')
		if len(token) > 1:
			token_dict = {feat.split('=')[0]:feat.split('=')[1] for feat in token[5:]}
			features.append(token_dict)
			labels.append(token[4])
	return features, labels


if __name__ == '__main__':

	model_file = sys.argv[1]
	vectorizer_file = sys.argv[2] 	

	train_features, y_train = load_data(sys.stdin)
	y_train = np.asarray(y_train)
	classes = np.unique(y_train)

	v = DictVectorizer()
	X_train = v.fit_transform(train_features)


	clf = MultinomialNB(alpha=0.01)

	#IF GRIDPARAM
	'''from sklearn.model_selection import GridSearchCV

	param_grid = {'alpha': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 10.0]} #default 0.01

	grid = GridSearchCV(clf, param_grid, cv=5)
	grid.fit(X_train, y_train)

	print(grid.best_params_)'''

	clf.partial_fit(X_train, y_train, classes)

	#Save classifier and DictVectorizer
	dump(clf, model_file) 
	dump(v, vectorizer_file)