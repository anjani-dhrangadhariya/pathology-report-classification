##################################################################################
# Imports
##################################################################################
# scikit learn imports
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import scale, StandardScaler, Normalizer, label_binarize
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score, classification_report, roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import PredefinedSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# visualization
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D

# Natural Language Toolkit
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
#nltk.download()
from nltk import ngrams, pos_tag

import numpy as np
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
import argparse
import pdb
import random
import collections, numpy
import json
import pandas as pd
import os
import glob
import itertools
import operator

# saving model
import shutil

# Import data getters
from data_builders.DocumentBuilder_ML import DocumentBuilder

# Visualization tools
from tensorboardX import SummaryWriter
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# saving the best models
saved_models = []
import pickle

##################################################################################
# Set all the seed values
##################################################################################
# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt

def get_param_grids(s):
    
    if 'logistic_regression' in s:
        
        param_grid = [{'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10)],
                      'vect__max_features': (None, 5000, 10000, 50000),
                      'vect__max_df': [0.7, 0.8, 0.9],
                      'vect__min_df': [0.0, 0.1, 0.2, 0.3, 0.4],
                      'vect__norm': ['l1','l2'],
                      'clf__penalty': ['l1','l2'],
                      'clf__class_weight': ['balanced', None]
              }
             ]
          
        return param_grid
    
    if 'svm' in s:      
  
        param_grid = [{'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10)],
                       'vect__max_features': (None, 5000, 10000, 50000),
                       'vect__max_df': [0.7, 0.8, 0.9],
                       'vect__min_df': [0.0, 0.1, 0.2, 0.3, 0.4],
                       'vect__norm': ['l1','l2']
                      }
                     ]
        
        return param_grid
    
    if 'knn' in s:
        
        param_grid = [{'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10)],
                        'vect__max_features': (None, 5000, 10000, 50000),
                        'vect__max_df': [0.7, 0.8, 0.9],
                        'vect__min_df': [0.0, 0.1, 0.2, 0.3, 0.4],
                        'vect__norm': ['l1','l2'],
                        'clf__n_neighbors': [9, 11, 21, 31],
                        'clf__metric': ['minkowski', 'euclidean']
              }
             ]
        
        return param_grid
    
    if 'trial' in s:
        
        param_grid = [{'clf__penalty': ['l2']
              }
             ]
        
        return param_grid
    
    else:
        raise ValueError('Please correctly specify the name of algorithm to apply...')


def get_classifier_pipeline(s):
    
    # Initialize vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words=None, smooth_idf=True)
    
    if 'logistic_regression' in s:
        
        log_reg_clf = LogisticRegression(intercept_scaling=1, random_state=42, solver='liblinear')
        log_reg_clf_tfidf = Pipeline([('vect', tfidf_vectorizer), ('scaler', normalizer), ('clf', log_reg_clf)])
        
        return log_reg_clf_tfidf
    
    if 'svm' in s:
        
        svm_clf = CalibratedClassifierCV(base_estimator=LinearSVC(penalty = 'l2', class_weight = 'balanced', fit_intercept=False, random_state=42, verbose=0, dual=False))
        svm_clf_tfidf = Pipeline([('vect', tfidf_vectorizer), ('norm', normalizer), ('clf', svm_clf)])
        
        return svm_clf_tfidf
    
    if 'knn' in s:
        
        knn_clf = KNeighborsClassifier(weights='uniform', algorithm='auto')
        knn_clf_tfidf = Pipeline([('vect', tfidf_vectorizer), ('norm', normalizer), ('clf', knn_clf)])
        
        return knn_clf_tfidf
    
    if 'trial' in s:
        
        svm_clf = CalibratedClassifierCV(base_estimator=LinearSVC(penalty='l2', fit_intercept=False, class_weight='balanced', random_state=42, verbose=1, dual=False))
        print(svm_clf.get_params().keys())
        svm_clf_tfidf = Pipeline([('vect', tfidf_vectorizer), ('norm', normalizer), ('clf', svm_clf)])
        
        return svm_clf_tfidf
        
    else:
        raise ValueError('Please correctly specify the name of algorithm to apply...')


def execute_baselines(s, X_train, X_test, y_train, y_test):
    
    train_roc_auc = []
    test_roc_auc = []

    precision_high = []
    recall_high = []
    f1_high = []

    precision_low = []
    recall_low = []
    f1_low = []

    print('-' * 30)
    print('Training set has ', list(y_train).count(1), ' low-grade instances and ', list(y_train).count(0), ' high-grade instances.')
    #print('Evaluation set has ', list(y_eval).count(1), ' low-grade instances and ', list(y_eval).count(0), ' high-grade instances.')
    print('Test set has ', list(y_test).count(1), ' low-grade instances and ', list(y_test).count(0), ' high-grade instances.')
    print('-' * 30)
    print('\n')


    # y_train = y_train.astype('int')

    clf = get_classifier_pipeline(s)
    params = get_param_grids(s)

    gridSearch = GridSearchCV(clf, params, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
    grid_result = gridSearch.fit(X_train, y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    train_roc_auc.append(grid_result.best_score_)

    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    # for mean, param in zip(means, params):
    #     print("Mean ROC-AUC score: %f with: %r" % (mean, param))

    print('-' * 30)
    filename = '/best_models/' + s + '_tfidf/' + 'finalized_model.sav'
    pickle.dump(gridSearch.best_estimator_, open(filename, 'wb'))
    y_pred = gridSearch.best_estimator_.predict(X_test)
    y_test = y_test.to_numpy()
    y_test = y_test.astype('int')

    ROCAUC_score = roc_auc_score(y_test, y_pred)
    print('The ROC-AUC score for the test set is: ', ROCAUC_score)
    test_roc_auc.append(ROCAUC_score)

    classReport =  classification_report(y_test, y_pred, output_dict = True)
    print(classReport)

    precision_high.append(classReport['1']['precision'])
    recall_high.append(classReport['1']['recall'])
    f1_high.append(classReport['1']['f1-score'])

    precision_low.append(classReport['0']['precision'])
    recall_low.append(classReport['0']['recall'])
    f1_low.append(classReport['0']['f1-score'])
        
    ## Print mean scores here
    print('ROC AUC SCORE: ', test_roc_auc)
    meanTrainPRU = sum(train_roc_auc)/len(train_roc_auc)
    meanTestPRU = sum(test_roc_auc)/len(test_roc_auc)

    print('Mean training ROC-AUC score is: ', meanTrainPRU)
    print('Mean test ROC-AUC score is: ', meanTestPRU)

    meanP = sum(precision_high)/len(precision_high)
    meanR = sum(recall_high)/len(recall_high)
    meanF1 = sum(f1_high)/len(f1_high)

    print('Mean precision for high-grade reports on the test set is: ', meanP)
    print('Mean recall for high-grade reports on the test set is: ', meanR)
    print('Mean F1 for high-grade reports on the test set is: ', meanF1)

    meanP_0 = sum(precision_low)/len(precision_low)
    meanR_0 = sum(recall_low)/len(recall_low)
    meanF1_0 = sum(f1_low)/len(f1_low)

    print('Mean precision for low-grade reports on the test set is: ', meanP_0)
    print('Mean recall for low-grade reports on the test set is: ', meanR_0)
    print('Mean F1 for low-grade reports on the test set is: ', meanF1_0)

    # Plot confusion matrix
    test_cm = confusion_matrix(y_test, y_pred, labels=list([0, 1]))
    plt.switch_backend('agg')
    xticklabels = yticklabels = ['low grade', 'high grade']
    f = sn.heatmap(test_cm, annot=True, annot_kws={"size": 30}, cmap='Blues', fmt='g', xticklabels=xticklabels, yticklabels=yticklabels) # font size 
    title = s + '\n' + 'tf-idf'
    plt.title(title)
    plt.savefig("/confusion_matrix/"+ s + "_tfidf/confusion_matrix" + ".png", dpi=400)
    f.clear()

    # Plot ROC_AUC curve
    # generate a no skill prediction (majority class)
    noskill_probs = [0 for _ in range(len(y_test))]
    ns_fpr, ns_tpr, _ = roc_curve(y_test, noskill_probs)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
    plt.plot(lr_fpr, lr_tpr, marker='.', label=s)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig("/roc_auc_curve/"+ s + "_tfidf/roc_auc_curve" + ".png", dpi=400)


if __name__ == "__main__":

    ##################################################################################
    # Get the data loader
    ##################################################################################

    normalizer = Normalizer()

    X_train_i, X_test, y_train_i, y_test = DocumentBuilder.get_data_loaders(augment = True)

    # baselines = ['logistic_regression', 'svm', 'knn']
    baselines = ['knn']

    for eachModel in baselines:
        print('#' * 50)
        print('Executing ', eachModel, ' pipeline (with augmentation)...')
        print('#' * 50)
        execute_baselines(eachModel, X_train_i, X_test, y_train_i, y_test)
