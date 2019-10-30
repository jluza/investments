# -*- coding: utf-8 -*-
from selenium import webdriver
from bs4 import BeautifulSoup
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

def Preparador(path, sample = 1):
    
    
    
    print()
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print()
    print('Levantando archivos en ' + path)
    print()
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print()
    
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
    acquisitions_history = pd.read_excel('acquisitions.xlsx', index_col = 0)
    companies_history = pd.read_excel('companies.xlsx', index_col = 0)
    
    companies_history.columns = ['Organization Name', 'Organization Name URL', 'Description', 'Categories',
       'Headquarters Location']
    
    comp = pd.concat([companies,companies1])
    companies_history = pd.concat([companies_history, comp], how = 'inner')
    companies_history.to_excel('companies_history.xlsx')
    comp.drop_duplicates(inplace = True)
    comp.set_index('Organization Name URL', inplace = True)    
    
    fr = fr.join(comp, how = 'left', rsuffix = '_right')
    fr['Transaction'] = 'Funding Round'
    fr['Entity'] = fr['Transaction Name URL'] + '#/entity'
    fr.drop('Organization Name_right', axis = 1, inplace = True)
    acq = acq.join(comp, how = 'left')
    acq['Transaction'] = 'Acquisition'
    acq['Entity'] = acq['Transaction Name URL'] + '#/entity'
    acq.drop(['Announced Date Precision', 'Organization Name'], axis = 1, inplace = True)

    m = pd.concat([fr,acq], join = 'outer', sort = True)
    m.drop('CB Rank (Company)', axis = 1, inplace = True)
        
    
    entity_database = list(set(fundings_history['Entity'].drop_duplicates().tolist() + acquisitions_history['Entity'].drop_duplicates().tolist()))
    m = m[~m['Entity'].isin(entity_database)]
    if sample < 1:
        m = m.sample(frac = sample, replace = False)
        
    
    
    return m, database, investments, class_hist

def Scrapeardeals(new):
    print()
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print()
    print('Scrapeando htmls de deals # (entity)')
    print()
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print()
    
    # Scrapeamos
    already_scraped = [] 
    links = new['Entity'].drop_duplicates()
    print('Cantidad de links: ' + str(len(links)))
    print()
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print()
    duration = 1000  # milliseconds
    freq = 440  # Hz
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    sources = []
    for i in links:
        time.sleep(2)
        print('Restantes: ' + str(len(links)-len(already_scraped)))
        driver = webdriver.Chrome(r"D:\Dropbox (MPD)\Analytics Argentina\Software and Tools\chromedriver_win32\chromedriver.exe", chrome_options = chrome_options)
        driver.get(i)
        source = driver.page_source
        while 'Please verify you are a human' in source: 
            winsound.Beep(freq, duration)
            key = input('Press ENTER')
            driver.get(i)
            source = driver.page_source
            break
    
        sources.append((i,source))
        already_scraped.append(i)
        driver.quit()
    return sources

def ParseandoHTMLdeals(sources):         # Sources: lista con los html de los deals
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print()
    print('Parseando html de los deals')
    print()
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    
        
    
    dato = []
    organizations = []
    for i in sources:

        print (i[0])
        deals = {}
        if 'funding_round' in i[0]:
            SOPA = BeautifulSoup(i[1], features = 'lxml')
            entity = SOPA.find_all('entity-section', {'class':"ng-star-inserted"})
            round_type = []
            lead_investors = 0
            multiple_investors = []
            links_mi = []
            try:
                inx = entity[0].find_all('a',{'class':'cb-link component--field-formatter field-type-identifier ng-star-inserted'})
                
                inv = 'https://www.crunchbase.com' + str(inx[0]['href'])
                organizations.append(inv.strip())
            except:
                pass
            
            try:
                for l in entity[2].find_all('div', {'class':"flex-no-grow cb-overflow-ellipsis identifier-label"}):
                    multiple_investors.append(l.get_text().strip())
            except:
                pass
            try:
                for l in entity[2].find_all('a', {'class':"cb-link component--field-formatter field-type-identifier ng-star-inserted"}):
                    links_mi.append(('https://www.crunchbase.com' + str(l['href']).strip())) 
            except:
                pass
            if len(multiple_investors)==0:
                links_mi = 'Not Disclosed'
                multiple_investors = 'Not Disclosed'
                lead_investors = 'Not Disclosed'
            else:
                li = []
                for L in entity[1].find_all('a',{'class':"cb-link"}):
                    li.append(L.get_text().strip())
                if len(li) != 0:
                    lead_investors = [', '.join(li) for x in range(len(links_mi))]
                else:
                    lead_investors = 'No lead investors'
                
                for x in range(len(links_mi)):
                    round_type.append(entity[0].find('span', {'class':"component--field-formatter field-type-enum ng-star-inserted"}).get_text().strip())

            if len(round_type)==0:
                round_type = 'Not Disclosed'
            deals['Deal #'] = [i[0] for x in range(len(multiple_investors))]
            deals['Round Type'] =  round_type
            deals['Multiple Investors'] = multiple_investors
            deals['Links Investors'] = links_mi
            deals['Lead Investors'] = lead_investors
            deals['Link Investee'] = [inv for x in range(len(multiple_investors))]
            
            for x in links_mi:
                if (type(x) != float) & (x.startswith('http')):
                    organizations.append(x)
            
        if 'acquisition' in i[0]:
            SOPA = BeautifulSoup(i[1], features = 'lxml')
            try:
                entity = SOPA.find_all('entity-section', {'class':"ng-star-inserted"})
                
                orgs = entity[0].find_all('a',{'class':"cb-link"})
                dats = []
                deals['Deal #'] = i[0]
                deals['Round Type'] = ['Acquisition']
                deals['Multiple Investors'] = ['Acquisition']
                deals['Lead Investors'] = 'Acquisition'
                for x in orgs:
                    # La primer organizacion en esta lista es el acquired, el segundo es el acquiring.
                    newurl = 'https://www.crunchbase.com' + x['href']
                    print(newurl)
                    organizations.append(newurl)
                    dats.append(newurl)    
                deals['Link Investee'] = dats[0].strip()
                deals['Links Investors'] = dats[1].strip()
            except:
                continue
    
        dato.append(deals)
    
    
    print()
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    
            
    
    organizations = list(set(organizations))
    return dato, organizations

def ScrapearORGS(organizations):
    print()
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    
    print()
    print('Numero de organizaciones a scrapear: ' + str(len(organizations)))
    duration = 1000  # milliseconds
    freq = 440  # Hz
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    print()
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    htmls_organizations = []
    for i in organizations:
        time.sleep(2)
        print('Restantes: ' + str(len(organizations) - len(htmls_organizations)))
        driver = webdriver.Chrome(r"D:\Dropbox (MPD)\Analytics Argentina\Software and Tools\chromedriver_win32\chromedriver.exe", chrome_options = chrome_options)
        driver.get(i)
        source = driver.page_source
        while 'Please verify you are a human' in source: 
            winsound.Beep(freq, duration)
            key = input('Press ENTER')
            driver.get(i)
            source = driver.page_source
            break
        htmls_organizations.append((i,source))
        driver.quit()
    return htmls_organizations

def ParsearHTMLOrganizaciones(htmls_organizations):   # Lista con los html de las organizaciones.
    print()
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print()
    print('Parseando html de las organizaciones')
    print()
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print()
    
    data = []
    for i in htmls_organizations:
        names = []
        investors_regions = []
        descriptions = []
        sop = BeautifulSoup(i[1], features = 'lxml')
        entities = sop.find_all('entity-section',{'class':"ng-star-inserted"})
        try:
            dat = entities[0].find('div', {'class':"flex layout-column layout-align-center-center layout-align-gt-sm-center-start text-content"})
        except:
            print('Error')
            continue
        try:
            investors_regions.append(dat.find_all('div',{'class':"ng-star-inserted"})[2].get_text().strip())
        except:
            investors_regions.append('Sin datos')    
        try:    
            dat = entities[0].find('div', {'class':"flex layout-column layout-align-center-center layout-align-gt-sm-center-start text-content"})
        except:
            descriptions.append('')
            print ('Sin datos')    
        try:
            names.append(dat.find_all('div',{'class':"ng-star-inserted"})[0].text.strip())
        except:
            names.append('Sin datos')
        try:    
            descriptions.append(dat.find_all('div',{'class':"ng-star-inserted"})[1].text.strip())
        except:
            descriptions.append('')
        try:    
            for P in entities[0].find_all('p'):
                descriptions.append(str(P.text).strip())
                #print (P.text)
        except:
            descriptions.append('Sin datos')
        descriptions=' '.join(descriptions)
        data.append([i[0].strip(),*names,*investors_regions,descriptions.strip()])
    
    return data

def Estructurador(dato, data, new): # Dato: lista con los deals. Data: scrapeo de las organizaciones.
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print()
    print('Estructurando los nuevos datos')
    print()
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    
        
    
    
    dict1 = {}
    for i in range(len(dato)):
        dict1[i] = pd.DataFrame(dato[i])
    
    datos = pd.concat(dict1, ignore_index = True, sort = False)
    companias = pd.DataFrame(data, columns = ['Link organization','Company Name','Location','Description'])
    
    datos = datos.reset_index(drop=True)
    
    mer = datos.merge(companias,how= 'left',left_on= 'Link Investee', right_on= 'Link organization')
    datos[['Investee Name','Geography','Description']] = mer[['Company Name', 'Location', 'Description']]
    datos = datos[['Deal #','Round Type','Money Raised', 'Lead Investors', 'Multiple Investors','Links Investors', 'Link Investee', 'Investee Name','Geography','Description']]
    
    
    mer2 = datos.merge(companias, how = 'left',left_on = 'Links Investors',right_on = 'Link organization')
    datos[['Company Name', 'Location', 'Investor Description']] = mer2[['Company Name', 'Location', 'Description_y']]
    datos = datos[['Deal #','Round Type','Money Raised', 'Lead Investors',
           'Multiple Investors', 'Links Investors','Company Name', 'Location', 'Investor Description', 'Link Investee', 'Investee Name', 'Geography',
           'Description']]
    


    datos['Category'] = datos.merge(new,how = 'left', left_on='Investee Name', right_on= 'Investee')['Category']
    
    datos['Date'] = pd.to_datetime(datos.merge(new,how = 'left', left_on='Deal #', right_on= 'Entity')['Announced Date'])
    
    datos['Week'] = datos['Date'].dt.year.astype('Int32').apply(str) + "-W" + datos['Date'].dt.week.astype('Int32').apply(str)
    datos['Quarter'] = datos['Date'].dt.quarter.apply(int) 
    datos['Year-Quarter'] = datos['Date'].dt.year.apply(str) + '-Q' + datos['Date'].dt.quarter.apply(str) 
    datos['Month'] = datos['Date'].dt.month
    datos['Month/Year'] =  datos['Date'].dt.month.apply(str) + '-' + datos['Date'].apply(lambda x: str(x.year))
    datos['Lead'] = datos.apply(lambda x: x['Multiple Investors'] in x['Lead Investors'], axis=1)
    datos['countInvestors'] = datos.groupby('Deal #')['Deal #'].transform('size')
    datos['countLeadInvestors'] = datos.apply(lambda x: len(x['Lead Investors'].split(',')), axis=1)
    datos['countNonLeadInvestors'] = datos['countInvestors'] - datos['countLeadInvestors']

    
    datos['Money Raised Currency (in USD)'] = datos.set_index('Entity').merge(new.set_index('Deal #')['Money Raised Currency (in USD)'], how = 'left' )
    datos['Amount (M USD)'] = pd.to_numeric(datos['Money Raised Currency (in USD)'], errors = 'coerce')
    datos['Amount (M USD)'] = datos['Amount (M USD)']/1000000
    
    datos.loc[datos['Lead'] == True, 'Investment Share'] = (0.6*datos['Amount (M USD)'].astype(float))/datos['countLeadInvestors']
    datos.loc[datos['Lead'] == False, 'Investment Share'] = (0.4*datos['Amount (M USD)'].astype(float))/datos['countNonLeadInvestors']
    datos.loc[datos['countInvestors'] == 1, 'Investment Share'] = (datos['Amount (M USD)'].astype(float))
    datos['Amount (M USD)'] = datos['Investment Share']
        
    datos['Investor Region'] = datos.apply(lambda x: str(x['Location']).split(', ')[-1] , axis=1)
    datos['Investee Region'] = datos.apply(lambda x: str(x['Geography']).split(', ')[-1] , axis=1)
    
    datos = datos.fillna('Not Disclosed').replace('nan','Not Disclosed')
    datos = datos.drop_duplicates()
    
    datos = datos.merge(new,how = 'left', left_on='Investee Name', right_on= 'Investee')
    datos['Investor'] = datos['Multiple Investors']
    datos['Investor Region2'] = datos['Investor Region']
    datos['Investee Region2'] = datos['Investee Region']

    datos['Round Type'] = datos['Round Type'].replace({'Series A' : '1 - First Round',
                                                   'Series B' : '2 - Second Round',
                                                   'Series C' : '3 - Third Round',
                                                   'Series D' : '4 - Fourth Round',
                                                   'Series E' : '5 - Fifth Round',
                                                   'Series F' : '6 - Sixth Round',
                                                   'Series G' : '7 - Seventh Round',
                                                   'Series H' : '8 - Eighth Round',
                                                   'Series I' : '9 - Ninth Round',
                                                   'Series J' : '10 - Tenth Round'})
    datos['Investor'] = datos['Multiple Investors']
    datos['Round Number'] = ''
    datos['Investors'] = datos['Multiple Investors']
    header_list = ['Link Investee', 'Investors', 'Multiple Investors', 'Links Investors', 
       'Investor Region', 'Investor Region2', 'Investee Name', 
      'Investee Region', 'Investee Region2', 'Amount (M USD)', 'Round Type', 'Pos Valuation/Market Cap', 'Area of Focus', 'Category.1',
       'Date', 'Deal #', 'Week', 'Month', 'Quarter',
       'Year-Quarter', 'Month/Year']
   
  
    master = datos.reindex(columns = header_list)
    changeIndex = master.loc[master['Multiple Investors'] == 'Acquisition'].index
    changeCompany = master.loc[master['Multiple Investors'] == 'Acquisition'].merge(companias, how = 'left', left_on = 'Links Investors', right_on = 'Link organization')['Company Name']
    change = pd.Series(changeCompany.tolist(), name = 'Multiple Investors', index = changeIndex.tolist())
    master.update(change)
    
    return master, datos, companias


def Clasificar(database, new, path):
    pd.options.mode.chained_assignment = None
    if 'Response by Category' in list(database.columns):
        database = database.drop(['Response by Category','Response by Description'], axis = 1)
    database = database.sample(frac= 0.4, replace = False)
    
    #Chequeo las companias que ya estaban clasificadas
    #d = new.merge(database, how ='left', left_on='Organization Name', right_on = 'Investee')[['Investee','Category.1','Area of Focus']]
    #new = new.merge(d, how = "left", left_on = "Organization Name", right_on = "Investee")
    #new = new.drop(columns=["Investee"])
    
    database["Category.1"] = database["Category.1"].replace("rejected", "Rejected")
    database["Category.1"] = database["Category.1"].replace("B2C ", "B2C")
    database["Category.1"] = database["Category.1"].replace("FIntech", "Fintech")

    database['Prediction'] = np.nan
    new['Prediction'] = np.nan
    new = new.drop(['Prediction'], axis=1)

    #CLASIFICADOR
    
    warnings.filterwarnings('ignore')
    
    
    print('Importando bases de datos')
    
    new = new.rename(columns = {'Categories':'Category','Organization Name':'Investee'})
    train = database[['Operation','Investee', 'Category', 'Description', 'Category.1', 'Area of Focus']].dropna()
    newdata = new[['Transaction Name','Investee', 'Category', 'Description']]
    
    
    print('Preprocesamiento del texto')
    
    stop_words = stopwords.words('english')
    
    for column in ['Category','Description']:
        
        train[column] = train[column].apply(lambda x: (" ".join(str(x).lower() for x in str(x).split())).encode('utf-8').decode('utf-8'))  # lower case
        train[column] = train[column].str.replace('[^\w\s]', ' ')          																											# removing punctuation
        train[column] = train[column].apply(lambda x: " ".join(str(x) for x in str(x).split() if x not in stop_words))   # removing stop words
        newdata[column] = newdata[column].apply(lambda x: (" ".join(x.lower() for x in str(x).split())))  # lower case
        newdata[column] = newdata[column].str.replace('[^\w\s]', ' ')																		# removing punctuation
        newdata[column] = newdata[column].apply(lambda x: " ".join(str(x) for x in str(x).split() if x not in stop_words))   # removing stop words
    
    
    train_src1 = train[['Category','Description','Category.1']]
    train_src1['Rejected?'] = 0
    train_src1.loc[train_src1['Category.1'] != 'Rejected', 'Rejected?'] = 1
    
    new_src1 = newdata[['Category','Description']]
    #new_src1['Rejected?'] = 0
    #new_src1.loc[new_src1['Category.1'] != 'Rejected', 'Rejected?'] = 1
    
    
    #Binarizacion
    vectorizer = CountVectorizer()
    
    vectorI = pd.DataFrame(vectorizer.fit_transform(train_src1['Category']).toarray())
    vectorI_new = pd.DataFrame(vectorizer.transform(new_src1['Category']).toarray())
    vectorIdes = pd.DataFrame(vectorizer.fit_transform(train_src1['Description']).toarray())
    vectorIdes_new = pd.DataFrame(vectorizer.transform(new_src1['Description']).toarray())
    
    vectorI = pd.concat([vectorI, vectorIdes], axis = 1)
    vectorI_new = pd.concat([vectorI_new, vectorIdes_new], axis = 1)
    
    print('Entrenamiento')
    
    #Clasificacion binaria: Rechazadas vs no rechazadas
                #Resampling + Random Forest
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
    brf.fit(vectorI, train_src1['Rejected?'])
    y_train_pred = brf.predict(vectorI)
    print('Confusion matrix: \n' , confusion_matrix(train_src1['Rejected?'], y_train_pred))
    print('Accuracy: \n' , accuracy_score(train_src1['Rejected?'], y_train_pred))
    print('Recall: \n' , recall_score(train_src1['Rejected?'], y_train_pred))
    
    
    print('Clasificacion y exportacion')
    #Ajustando modelo a nuevos datos
    y_new_predict = brf.predict(vectorI_new)
    y_new_predict_proba = brf.predict_proba(vectorI_new)
    
    newdata['Prediction'] = y_new_predict
    newdata['Prob. of being rejected'] = y_new_predict_proba[:,0]
    newdata['Prob. of being of interest'] = y_new_predict_proba[:,1]
    

    
    #Creamos archivo Companies y exportamos
    new = pd.concat([new, newdata[['Prediction','Prob. of being rejected','Prob. of being of interest']]], axis=1, sort=False) 

    return new

def Estructurador1(dato, data, new): # Dato: lista con los deals. Data: scrapeo de las organizaciones.
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print()
    print('Estructurando los nuevos datos')
    print()
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    
    
    dict1 = {}
    for i in range(len(dato)):
        dict1[i] = pd.DataFrame(dato[i])
    
    datos = pd.concat(dict1, ignore_index = True)
    companias = pd.DataFrame(data, columns = ['Link organization','Company Name','Location','Description'])
    
    datos = datos.reset_index(drop=True)
    mer = datos.merge(companias,how= 'left',left_on= 'Link Investee', right_on= 'Link organization')
    datos[['Investee Name','Geography','Description']] = mer[['Company Name', 'Location', 'Description']]
    datos = datos[['Deal #','Round Type','Money Raised', 'Lead Investors', 'Multiple Investors','Links Investors', 'Link Investee', 'Investee Name','Geography','Description']]
    
    
    mer2 = datos.merge(companias, how = 'left',left_on = 'Links Investors',right_on = 'Link organization')
    datos[['Company Name', 'Location', 'Investor Description']]=mer2[['Company Name', 'Location', 'Description_y']]
    datos = datos[['Deal #','Round Type','Money Raised', 'Lead Investors',
           'Multiple Investors', 'Links Investors','Company Name', 'Location', 'Investor Description', 'Link Investee', 'Investee Name', 'Geography',
           'Description']]
    datos['Lead'] = datos.apply(lambda x: x['Multiple Investors'] in x['Lead Investors'], axis=1)
    datos['countInvestors'] = datos.groupby('Deal #')['Deal #'].transform('size')
    datos['countLeadInvestors'] = datos.apply(lambda x: len(x['Lead Investors'].split(',')), axis=1)
    datos['countNonLeadInvestors'] = datos['countInvestors'] - datos['countLeadInvestors']
    print(datos.merge(new, how = 'left', left_on = 'Deal #', right_on = 'Entity').columns)
    datos['Money Raised Currency (in USD)'] = datos.set_index('Entity').merge(new.set_index('Deal #')['Money Raised Currency (in USD)'], how = 'left' )
    datos['Amount (M USD)'] = datos['Money Raised Currency (in USD)']/1000000
    
    datos.loc[datos['Lead'] == True, 'Investment Share'] = (0.6*datos['Amount (M USD)'].astype(float))/datos['countLeadInvestors']
    datos.loc[datos['Lead'] == False, 'Investment Share'] = (0.4*datos['Amount (M USD)'].astype(float))/datos['countNonLeadInvestors']
    datos.loc[datos['countInvestors'] == 1, 'Investment Share'] = (datos['Amount (M USD)'].astype(float))
    datos['Amount (M USD)'] = datos['Investment Share']
    return datos