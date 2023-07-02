import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from utils import *
from classifiers import *
from preprocess import  *

seed = 42
random.seed(seed)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", 
                        help="Input data in csv format", type=str)
    parser.add_argument("-v", "--voc_size", 
                        help="Vocabulary size", type=int)
    parser.add_argument("-a", "--analyzer",
                         help="Tokenization level: {word, char}", 
                        type=str, choices=['word','char'])
    return parser

if __name__ == "__main__":
    #parser = get_parser()
    #args = parser.parse_args()
    
    #raw = pd.read_csv(args.input)
    raw = pd.read_csv('data\dataset.csv')
    print(raw.shape)
    analyzer = "word"
    voc_size = 1000
    
    # Languages
    languages = set(raw['language'])
    element_counts = raw['language'].value_counts()
    
    
    print('========')
    print('Languages', languages)
    print('========')
    
    print('========')
    print("Language Count: \n", element_counts)
    print('========')

    # Split Train and Test sets
    X=raw['Text']
    y=raw['language']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    print('========')
    print('Split sizes:')
    print('Train:', len(X_train))
    print('Test:', len(X_test))
    print('========')
    
    # Preprocess text (Word granularity only)
    if analyzer == 'word':
        X_train, y_train = preprocess_splitting_four(X_train,y_train)
        X_test, y_test = preprocess_splitting_four(X_test,y_test)

    #Compute text features
    features, X_train_raw, X_test_raw = compute_features(X_train, 
                                                            X_test, 
                                                            analyzer=analyzer, 
                                                            max_features=voc_size)

    print('========')
    print('Number of tokens in the vocabulary:', len(features))
    print('Coverage: ', compute_coverage(features, X_test.values, analyzer=analyzer))
    print('========')


    #Apply Naive Bayes Classifier  
    X_train, X_test = normalizeData(X_train_raw, X_test_raw)
    
    y_predict = applyNaiveBayes(X_train, y_train, X_test)
    
    print('========')
    print('NaiveBayes Results')
    print('Prediction Results:')    
    plot_F_Scores(y_test, y_predict)
    print('========')
    
    plot_Confusion_Matrix(y_test, y_predict, "Greens")
    
    #Apply kNN Classifier

    y_predict_knn = applyNearestNeighbour(X_train, y_train, X_test)
    
    print('========')
    print('kNN Results')
    print('Prediction Results:')
    plot_F_Scores(y_test, y_predict_knn)
    print('========')
    plot_Confusion_Matrix(y_test, y_predict_knn, "Purples")
    
    #Apply SVM Classifier

    """ y_predict_svm = applySVM(X_train, y_train, X_test)
    
    print('========')
    print('SVM Results')
    print('Prediction Results:')
    plot_F_Scores(y_test, y_predict_svm)
    print('========')
    plot_Confusion_Matrix(y_test, y_predict_svm, "cool") """
    
    #Apply Decision Tree Classifier

    y_predict_dectree = applyDecisionTree(X_train, y_train, X_test)
    
    print('========')
    print('Decision Tree Results')
    print('Prediction Results:')
    plot_F_Scores(y_test, y_predict_dectree)
    print('========')
    plot_Confusion_Matrix(y_test, y_predict_dectree, "Oranges")
    
    #Plot PCA
    print('========')
    print('PCA and Explained Variance:')
    plotPCA(X_train, X_test, y_test, languages) 
    print('========')