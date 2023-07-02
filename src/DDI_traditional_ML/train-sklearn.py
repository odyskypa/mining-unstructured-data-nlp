#!/usr/bin/env python3

import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import numpy as np
import argparse
from joblib import dump


def load_data(data):
	features = []
	labels = []
	for interaction in data:
		interaction = interaction.strip()
		interaction = interaction.split('\t')
		interaction_dict = {feat.split('=')[0]:feat.split('=')[1] for feat in interaction[1:] }
		features.append(interaction_dict)
		labels.append(interaction[0])
	return features, labels


if __name__ == '__main__':

	model_file = sys.argv[1]
	vectorizer_file = sys.argv[2] 	

	train_features, y_train = load_data(sys.stdin)
	y_train = np.asarray(y_train)
	classes = np.unique(y_train)

	v = DictVectorizer()
	X_train = v.fit_transform(train_features)

	# define parameter grid for SVM
	param_grid = {
		'C': [0.01, 0.1, 1, 10],
		'penalty': ['l1', 'l2']
	}

	# initialize SVM classifier
	clf = LinearSVC()

	# perform grid search to find best hyper parameters
	grid_search = GridSearchCV(clf, param_grid, cv=5)
	grid_search.fit(X_train, y_train)

	# get best classifier from grid search
	clf = grid_search.best_estimator_

	clf.partial_fit(X_train, y_train, classes)

	#Save classifier and DictVectorizer
	dump(clf, model_file) 
	dump(v, vectorizer_file)