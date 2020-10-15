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

# required imports
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim.models.doc2vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

import multiprocessing

from collections import namedtuple
from collections import OrderedDict

from numpy import array

cores = multiprocessing.cpu_count()
#assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be SUPER slow otherwise"

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

# Libraries for imbalanced data
# import imblearn
# from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SMOTENC, SVMSMOTE
# from imblearn.keras import BalancedBatchGenerator
# from imblearn.pipeline import Pipeline as imbPipeline

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

def get_param_grids_d2v(s):
    
    if 'logistic_regression' in s:
        
        param_grid = [{'clf__penalty': ['none','l2'],
                      'clf__class_weight': ['balanced', None]
              }
             ]
          
        return param_grid
    
    if 'svm' in s:      
  
        param_grid = [{'clf__method': ['sigmoid']
              }
             ]
        
        return param_grid
    
    if 'knn' in s:
        
        param_grid = [{'clf__n_neighbors': [9, 11, 21, 31],
                       'clf__metric': ['minkowski', 'euclidean']
              }
             ]
        
        return param_grid
    
    if 'dtc' in s:
        
        param_grid = [{'clf__criterion': ['gini', 'entropy'],
                       'clf__max_depth': [3, 5, 7, None]
              }
             ]
        
        return param_grid
    
    if 'ada' in s:
        
        # , LogisticRegression(class_weight='balanced' ,solver='lbfgs', random_state=42)
        param_grid = [{'clf__base_estimator': [DecisionTreeClassifier(max_depth=1)],
                       'clf__n_estimators': [10, 20, 30, 50]
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


def get_classifier_pipeline_d2v(s):
    
    
    if 'logistic_regression' in s:
        
        log_reg_clf = LogisticRegression(intercept_scaling=1, solver='lbfgs', random_state=42)
        log_reg_clf_tfidf = Pipeline([('norm', normalizer), ('clf', log_reg_clf)])
        
        return log_reg_clf_tfidf
    
    if 'svm' in s:
        
        svm_clf = CalibratedClassifierCV(base_estimator=LinearSVC(penalty = 'l2', class_weight = 'balanced', fit_intercept=False, random_state=42, verbose=1, dual=False))
        svm_clf_tfidf = Pipeline([('norm', normalizer), ('clf', svm_clf)])
        
        return svm_clf_tfidf
    
    if 'knn' in s:
        
        knn_clf = KNeighborsClassifier(weights='uniform', algorithm='auto')
        knn_clf_tfidf = Pipeline([('norm', normalizer), ('clf', knn_clf)])
        
        return knn_clf_tfidf
    
    if 'dtc' in s:
        DT_clf = DecisionTreeClassifier(class_weight='balanced', splitter='best', min_samples_split=2, min_samples_leaf=1, random_state=42)
        DT_clf_tfidf = Pipeline([('norm', normalizer), ('clf', DT_clf)])
        
        return DT_clf_tfidf
    
    if 'ada' in s:
        
        ada_clf = AdaBoostClassifier(random_state=42)
        ada_clf_tfidf = Pipeline([('norm', normalizer), ('clf', ada_clf)])
        
        return ada_clf_tfidf
        
    else:
        raise ValueError('Please correctly specify the name of algorithm to apply...')

def labelledDocs(df):
    
    docs = []
    
    taggedDocument = namedtuple('taggedDocument', 'words tags')
    
    for index, row in df.iterrows():
        docs.append(taggedDocument(row['text'].split(), [row['category'], 'DOC_%s' % index]))
        
    return docs

def inferDocVecs(modelName, train_docs):
    
    train_vectors = []
    
    for eachDoc in train_docs:
        vector = modelName.infer_vector(eachDoc)
        train_vectors.append(vector)
        
    return train_vectors

def tokenize(df):
    
    test_df = []
    test_cat_df = []
    
    for index, row in df.iterrows():
        test_df.append(row['text'].split())
        test_cat_df.append(row['category'])
    
    return test_df, test_cat_df

def doc2vec_models(vec_size, epoch, window):
    
    
    simple_models = dict()
    
    # PV-DBOW plain
    simple_models['pvdbow'] = Doc2Vec(dm=0, vector_size=vec_size, negative=5, hs=0,  window=window, min_count=2, sample=0, workers=cores, epochs = epoch)
    
    # PV-DM w/ default averaging; a higher starting alpha may improve CBOW/PV-DM modes
    simple_models['pvdm'] = Doc2Vec(dm=1, vector_size=vec_size, window=window, negative=5, hs=0, min_count=2, sample=0, workers=cores, alpha=0.05, comment='alpha=0.05', epochs = epoch)
    
    # PV-DM w/   - big, slow, experimental mode
    # window=5 (both sides) approximates paper's apparent 10-word total window size
    simple_models['pvdmc'] = Doc2Vec(dm=1, dm_concat=1, vector_size=vec_size, window=window, negative=5, hs=0, min_count=2, sample=0, workers=cores, epochs = epoch)
    
    return simple_models


def doc2vec_documentlevel(s, X_train_i, X_test, y_train_i, y_test):
    
    best_model_vec_combination = dict()
    
    # Doc2vec parameters
    vec_size = [100, 300, 500]
    epochs = [20, 30, 50]
    window_sizes = [2, 3, 5]
    params = [vec_size, epochs, window_sizes]
    param_list = list(itertools.product(*params))
    
    # X_train, X_test, y_train, y_test = train_test_split(df_data['text'], df_data['category'], test_size=0.20, shuffle=True, random_state=42)

    df_train = pd.DataFrame({'text':X_train_i, 'category':y_train_i})
    df_test = pd.DataFrame({'text':X_test, 'category':y_test})
   
    # Tagged training docs
    taggedDocs = labelledDocs(df_train)
    
    # Tokenize test documents
    test_df, test_cat_df = tokenize(df_test)
    df_test = pd.DataFrame({'text':test_df, 'category':test_cat_df})
    
    # Tokenize train documents (train logistic regression using dataset)
    train_df, train_cat_df = tokenize(df_train)
    df_train = pd.DataFrame({'text':train_df, 'category':train_cat_df})  

    for eachParamTuple in param_list:

       
        ind_vec_size = eachParamTuple[0]
        ind_epoch = eachParamTuple[1]
        ind_window_size = eachParamTuple[2]
        
        allModels = doc2vec_models(ind_vec_size, ind_epoch, ind_window_size)
        for model_key, individualModel in allModels.items():
            if '+' not in model_key:
                individualModel.build_vocab(taggedDocs)
                print("%s vocabulary scanned & state initialized" % individualModel)

        print('\n\n')
        counter = 1
        for model_key, individualModel  in allModels.items():

            train_roc_auc = []
            test_roc_auc = []

            precision_high = []
            recall_high = []
            f1_high = []

            precision_low = []
            recall_low = []
            f1_low = []

            print('#' * 30)
            print('Doc2vec_', counter, ' - model name: ', model_key, ' params: ', str(eachParamTuple))
            print('#' * 30)
            counter = counter + 1

            individualModel.train(taggedDocs, total_examples=len(taggedDocs), epochs = ind_epoch)

            X_train_set = inferDocVecs(individualModel, df_train['text'])
            Y_train_set = df_train['category']

            ## Gridsearch logistic regression and SVM with doc2vec document level vectors
            clf = get_classifier_pipeline_d2v(s)
            params = get_param_grids_d2v(s)

            gridSearch = GridSearchCV(clf, params, scoring='roc_auc', cv=10, verbose=1, n_jobs=-15)

            grid_result = gridSearch.fit(X_train_set, Y_train_set)

            # summarize results
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            train_roc_auc.append(grid_result.best_score_)

            means = grid_result.cv_results_['mean_test_score']
            params = grid_result.cv_results_['params']

            filename = '/best_models/'+ s + '_docvec/' + str(individualModel).replace('/', '') + '_noisy_reports_finalized_model.sav'
            pickle.dump(gridSearch.best_estimator_, open(filename, 'wb'))
           

            X_test_set = inferDocVecs(individualModel, df_test['text'])
            Y_test_set = df_test['category']
            Y_test_set = Y_test_set.astype('int')

            y_pred = gridSearch.predict(X_test_set)
            y_pred = y_pred.astype('int')

            ROCAUC_score = roc_auc_score(y_test, y_pred)
            print('The ROC-AUC score for the test set is: ', ROCAUC_score)
            test_roc_auc.append(ROCAUC_score)
            
            key = str(s) + '_' + str(model_key) + '_' + str(eachParamTuple)
            best_model_vec_combination[key] = ROCAUC_score


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
            title = s +  '\n' + str(individualModel).replace('/', '')
            plt.title(title)
            plt.savefig("/confusion_matrix/"+ s + '_docvec/' + str(individualModel).replace('/', '') + '.png', dpi=400)
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
            plt.savefig("/roc_auc_curve/"+ s + '_docvec/' + str(individualModel).replace('/', '') + '.png', dpi=400)

            print('---------------------------------------------------------------')
            
    return best_model_vec_combination


if __name__ == "__main__":

    ##################################################################################
    # Get the data loader
    ##################################################################################

    # Doc2vec parameters
    # vec_size = [100, 300, 500]
    # epochs = [20, 30, 50]
    # window_sizes = [2, 3, 5]
    vec_size = [300]
    epochs = [20]
    window_sizes = [5]
    params = [vec_size, epochs, window_sizes]
    param_list = list(itertools.product(*params))

    normalizer = Normalizer()

    X_train_i, X_test, y_train_i, y_test = DocumentBuilder.get_data_loaders(augment = True)

    baselines = ['logistic_regression', 'svm', 'knn']

    best_model_vec = dict()

    for eachModel in baselines:

        print('#' * 50)
        print('Executing ', eachModel, ' pipeline...')
        print('#' * 50)
        best_model_vec_i = doc2vec_documentlevel(eachModel, X_train_i, X_test, y_train_i, y_test)
        
        key = max(best_model_vec_i.items(), key=operator.itemgetter(1))[0]
        value = max(best_model_vec_i.items(), key=operator.itemgetter(1))[1]
        
        best_model_vec[key] = value

        print(best_model_vec)