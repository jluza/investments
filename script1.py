import pandas as pd
import numpy as np
import os

direct = r'D:\Dropbox (MPD)\Analytics Argentina\Non-Billable projects\Investment Tracker'


fundings = []
companies = []
acquisitions = []

for x,y,z in os.walk(direct):
    for file in z:
        path = os.path.join(x,file)
        print(path)
        if file.startswith('funding-rounds-') & file.endswith('.csv'):
            print(file)
            fr = pd.read_csv(path)
            fundings.append(fr)

        if file.startswith('acquisitions-') & file.endswith('.csv'):
            print(file)
            acq = pd.read_csv(path)
            acquisitions.append(acq)
        if file.startswith('companies-'):
            print(file)
            comp = pd.read_csv(path)
            companies.append(comp)
        if (file.startswith('companies-') & file.endswith('(1).csv')):
            print(file)
            comp1= pd.read_csv(path)
            companies.append(comp1)

fundings = pd.concat(fundings)
fundings = fundings[['Funding Type', 'Money Raised',
         'Transaction Name', 'Transaction Name URL',
         'Organization Name', 'Organization Name URL',
         'Money Raised Currency', 'Money Raised Currency (in USD)',
         'Announced Date']]
fundings.reset_index(drop = True, inplace = True)
fundings = fundings.dropna(thresh = 5)

acquisitions = pd.concat(acquisitions)
acquisitions['Acquiree Name'].fillna(acquisitions['Acquired Company Name'],inplace = True)
acquisitions['Acquiree Name'].fillna(acquisitions['Acquired Organization Name'],inplace = True)
acquisitions['Acquirer Name'].fillna(acquisitions['Acquiring Organization Name'],inplace = True)
acquisitions['Acquirer Name'].fillna(acquisitions['Acquiring Company Name'],inplace = True)
acquisitions['Acquiree Name URL'].fillna(acquisitions['Acquired Company Name URL'],inplace = True)
acquisitions['Acquiree Name URL'].fillna(acquisitions['Acquired Organization Name URL'],inplace = True)
acquisitions['Acquirer Name URL'].fillna(acquisitions['Acquiring Organization Name URL'],inplace = True)
acquisitions['Acquirer Name URL'].fillna(acquisitions['Acquiring Company Name URL'],inplace = True)
acquisitions = acquisitions[['Transaction Name','Transaction Name URL','Acquiree Name','Acquiree Name URL','Acquirer Name','Acquirer Name URL','Announced Date']]

companies = pd.concat(companies)
companies['Company Name'].fillna(companies['Organization Name'], inplace = True)
companies['Company Name URL'].fillna(companies['Organization Name URL'], inplace = True)
companies['Categories'].fillna(companies['Category Groups'], inplace = True)
companies = companies[['Company Name', 'Company Name URL',
                       'Description','Categories','Headquarters Location']]

fundings.drop_duplicates(inplace = True)
acquisitions.drop_duplicates(inplace = True)
companies.drop_duplicates(inplace = True)

os.chdir(r'D:\Dropbox (MPD)\Analytics Argentina\Non-Billable projects\Investment Tracker\remake')
fundings.to_excel('fundings.xlsx')
acquisitions.to_excel('acquisitions.xlsx')
companies.to_excel('companies.xlsx')
