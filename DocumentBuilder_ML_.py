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




# sklearn
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Visualization tools
from tensorboardX import SummaryWriter
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# General imports
import pickle as pkl 
import json
import pandas as pd
import numpy as np
import os
import glob
import itertools
import operator
import collections
import csv
import random
import wimpy
from pathlib import Path
import xlrd

# sklearn
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight

# visualization
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
#import umap

# Text augmentation tools
from googletrans import Translator

# Natural Language Toolkit
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
#print(stop_words)
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk import ngrams, pos_tag
nltk.download('punkt')
# add punctuation marks to the stopword set
stop_words.add(';')
stop_words.add(':')
stop_words.add(',')
stop_words.add('#')
stop_words.add('(')
stop_words.add(')')
stop_words.add('report')
stop_words.add('electronically')
stop_words.add('signed')
stop_words.add('out')
stop_words.add('***')
stop_words.add('reviewed')
stop_words.add('approved')
stop_words.add('\'')
stop_words.add('\'\'')
stop_words.add('&')
stop_words.add('page')
stop_words.add('.')
stop_words.add('-')
stop_words.add('?')
stop_words.add(' ')
stop_words.add('--')
stop_words.add('_')
stop_words.add('  ')
stop_words.add('"')
stop_words.add('/')
stop_words.add('`')
stop_words.add('!')
stop_words.add(']')
stop_words.add('[')

# pyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# keras essentials
#from keras.preprocessing.sequence import pad_sequences

# transformer essentials
from transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification

# tensorflow modules
#import tensorflow as tf

# output formatting essentials
from tqdm import tqdm, trange
import io

##################################################################################
# Set all the seed values
##################################################################################
# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)


class DocumentBuilder:

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.MAX_LEN = MAX_LEN
        

    def get_data_loaders(augment = False):

        def augmentReport(reports, labels):

            print('Augmenting the reports through backtranslation process...')

            translator = Translator()

            # augment the report 54 times if it belongs to minority case
            # May be augment the training set to have balanced low-grade and high-grade reports
            count = collections.Counter(labels)
            majority_class = count[0]
            minority_class = count[1]

            minority_class_indices = []
            for i, element in enumerate( labels):
                if element == 1:
                    minority_class_indices.append(i)

            for i in range(minority_class, majority_class):
                randomly_chosen_index = (random.choice(minority_class_indices))
                report2augment = reports[randomly_chosen_index]

               
                # translate to German
                german_report = translator.translate(report2augment, dest='de')

                # backtranslate to English
                english_report = translator.translate(german_report.text, dest='en')

                reports.append(english_report.text)
                labels.append(1) 

            return reports, labels

        # load the reports here
        reportDir = '/reports_denoised/'

        # load report labels here
        def get_labels():
            workbook = xlrd.open_workbook('/nationwidechildrens.org_clinical_patient_prad_gleason.xlsx', on_demand=True)
            sheet = workbook.sheet_by_name('nationwidechildrens.org_clinica')
            # read header values into the list    
            keys = [sheet.cell(0, col_index).value for col_index in range(sheet.ncols)]

            report2grade = dict()

            for row_index in range(1, sheet.nrows):
                d = {keys[col_index]: sheet.cell(row_index, col_index).value
                    for col_index in range(sheet.ncols)}
                
                reportName = d['bcr_patient_barcode']
                reportScore = d['gleason_score']
                
                if reportScore == 6.0 or reportScore == 7.0:
                    report2grade[reportName] = 0 # low-grade
                else:
                    report2grade[reportName] = 1 # high-grade
            return report2grade

        report2grade = get_labels()
       
        all_reports = []
        all_labels = []

        count = 0
        # iterate the directory
        for filename in os.listdir(reportDir):

            count = count + 1
           
            fileStats = Path(reportDir+filename).stat()

            if fileStats.st_size > 3: # If the file is 
                with open(reportDir+filename, 'r') as pathReportFile:

                    buildReport = []

                    for line in pathReportFile:
                        if len(line) >= 5:

                            # tokenize, remove stop words and lowcase the text
                            tokenized_words = word_tokenize(line.rstrip().lower())

                            # tokens_without_sw = [word for word in tokenized_words]
                            tokens_without_sw = [word for word in tokenized_words if not word in stop_words]

                            # print(tokens_without_sw)

                            buildReport.append(' '.join(tokens_without_sw))


                    buildReport = ' '.join(buildReport)
                    all_reports.append(buildReport)
                    all_labels.append( report2grade[filename[0:12]] ) 

        # Total reports before augmentation
        print('Number high-grade before augmentation: ', all_labels.count(1))
        print('Number low-grade before augmentation: ', all_labels.count(0))

        #create a dataframe from the report chunks and the labels
        df_data = pd.DataFrame(
            {'text':all_reports,
            'category': all_labels
            })

        # get training_i and test datasets
        X_train_i, X_test, y_train_i, y_test = train_test_split(df_data['text'], df_data['category'], test_size=0.20, shuffle=True, random_state=42)               
        


        if augment == True:
            # augment them
            augmentedReports, augmentedLabels = augmentReport(list(X_train_i), list(y_train_i))

            print('Total number of reports loaded: ', len(augmentedReports) + len(X_test))
            print('Total number of reports for the class High-Grade after augmentation: ', list(augmentedLabels).count(1) + list(y_test).count(1))
            print('Total number of reports for the class Low-Grade after augmentation: ', list(augmentedLabels).count(0) + list(y_test).count(0))

            return augmentedReports, X_test, augmentedLabels, y_test
        else:
            return X_train_i, X_test, y_train_i, y_test