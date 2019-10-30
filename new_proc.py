# -*- coding: utf-8 -*-
from selenium import webdriver
from bs4 import BeautifulSoup
import logging
import numpy as np
import pandas as pd
import os
import time
import winsound
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import warnings
from builtins import str
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier
from selenium.webdriver.chrome.options import Options



path = r'D:\Dropbox (MPD)\Analytics Argentina\Non-Billable projects\Investment Tracker\2019-10-25 - Copy'
os.chdir(path)
logging.basicConfig(filename='investments.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

logging.info('Starting Investments')

for file in os.listdir(path):
    
    if file.startswith('funding-rounds-') & file.endswith('.csv'):
        print(file)
        fr = pd.read_csv(os.path.join(path, file))
        fr.set_index('Organization Name URL', inplace = True)
    if file.startswith('Companies'):
        print(file)
        database= pd.read_excel(os.path.join(path, file), sheet_name = 'Sheet1')
        
    if file.startswith('acquisitions-') & file.endswith('.csv'):
        print(file)
        acq = pd.read_csv(os.path.join(path, file))
        acq.rename(columns = {'Acquiree Name URL':'Organization Name URL'}, inplace = True)
        
        acq.set_index('Organization Name URL', inplace = True)
    if file.startswith('companies-'):
        print(file)
        companies = pd.read_csv(os.path.join(path, file))
    if (file.startswith('companies-') & file.endswith('(1).csv')):
        print(file)
        companies1= pd.read_csv(os.path.join(path, file))
    if file.startswith('Investments ') & (file.endswith('.xlsx')):
        print(file)
        investments = pd.read_excel(os.path.join(path, file))
    if file.startswith('class_history') & file.endswith('.xlsx'):
        print(file)
        class_hist = pd.read_excel(os.path.join(path, file))

fundings_history = pd.read_excel('fundings.xlsx', index_col = 0)
fundings_history.dropna(inplace = True, thresh = 2)

acquisitions_history = pd.read_excel('acquisitions.xlsx', index_col = 0)
acquisitions_history.dropna(inplace = True, thresh = 2)

companies_history = pd.read_excel('companies.xlsx')
companies_history.dropna(inplace = True, thresh = 2)
companies_history.drop_duplicates('Company Name', inplace = True)

companies_history.columns = ['Organization Name', 'Organization Name URL', 'Description', 'Categories',
   'Headquarters Location']

comp = pd.concat([companies,companies1])
companies_history = pd.concat([companies_history, comp], join = 'inner')
#companies_history.to_excel('companies_history.xlsx')
companies_history.drop_duplicates('Organization Name', keep = 'last', inplace = True)
companies_history.set_index('Organization Name URL', inplace = True)

comp.set_index('Organization Name URL', inplace = True)
fr = fr.join(comp, how = 'left', rsuffix = '_right')

fr['Transaction'] = 'Funding Round'
fr['Entity'] = fr['Transaction Name URL'] + '#/entity'
fr.drop('Organization Name_right', axis = 1, inplace = True)

acq = acq.join(comp, how = 'left')
acq['Transaction'] = 'Acquisition'
acq['Entity'] = acq['Transaction Name URL'] + '#/entity'
acq.drop(['Announced Date Precision', 'Organization Name'], axis = 1, inplace = True)

entity_database = list(set(fundings_history['Entity'].drop_duplicates().tolist() + acquisitions_history['Entity'].drop_duplicates().tolist()))

fr = fr[~fr['Entity'].isin(entity_database)]
fundings_history = pd.concat([fundings_history,fr], join = 'inner')
#fundings_history.to_excel('fundings_history.xlsx')

logging.info('Funding Rounds: %s',fr.shape[0])

acq = acq[~acq['Entity'].isin(entity_database)]
acquisitions_history = pd.concat([acquisitions_history,acq], join = 'inner')
#acquisitions_history.to_excel('acquisitions_history.xlsx')
logging.info('Acquisitions: %s',acq.shape[0])

class_hist.set_index('Investee', inplace = True)
fr.set_index('Organization Name', inplace = True)
acq.set_index('Acquiree Name', inplace = True)

fr = fr.join(class_hist[['Category.1','Area of Focus']], how = 'left')
acq = acq.join(class_hist[['Category.1','Area of Focus']], how = 'left')

clas_fr = fr.loc[fr['Category.1'].isna()]
logging.info('Funding Rounds to classify: %s',clas_fr.shape[0])

clas_acq = acq.loc[acq['Category.1'].isna()]
logging.info('Acquisitions to classify: %s',clas_acq.shape[0])

