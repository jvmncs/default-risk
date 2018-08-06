#! /usr/bin/python3
'''

    For speed improvements install and run with PyPy

    $ pypy3 main.py

    

'''


# imports
import sklearn as sk
import numpy as np
import pandas as pd
from pylab import *
from sklearn.preprocessing import Imputer,scale
from scipy.stats import chi2_contingency
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

#############################################################################
#                                                                           #
#                       APPLICATION COLUMN HEADERS                          #
#                                                                           #
# 'SK_ID_CURR' 'NAME_CONTRACT_TYPE' 'CODE_GENDER' 'FLAG_OWN_CAR'            #
#  'FLAG_OWN_REALTY' 'CNT_CHILDREN' 'AMT_INCOME_TOTAL' 'AMT_CREDIT'         #
#  'AMT_ANNUITY' 'AMT_GOODS_PRICE' 'NAME_TYPE_SUITE' 'NAME_INCOME_TYPE'     #
#  'NAME_EDUCATION_TYPE' 'NAME_FAMILY_STATUS' 'NAME_HOUSING_TYPE'           #
#  'REGION_POPULATION_RELATIVE' 'DAYS_BIRTH' 'DAYS_EMPLOYED'                #
#  'DAYS_REGISTRATION' 'DAYS_ID_PUBLISH' 'OWN_CAR_AGE' 'FLAG_MOBIL'         #
#  'FLAG_EMP_PHONE' 'FLAG_WORK_PHONE' 'FLAG_CONT_MOBILE' 'FLAG_PHONE'       #
#  'FLAG_EMAIL' 'OCCUPATION_TYPE' 'CNT_FAM_MEMBERS' 'REGION_RATING_CLIENT'  #
#  'REGION_RATING_CLIENT_W_CITY' 'WEEKDAY_APPR_PROCESS_START'               #
#  'HOUR_APPR_PROCESS_START' 'REG_REGION_NOT_LIVE_REGION'                   #
#  'REG_REGION_NOT_WORK_REGION' 'LIVE_REGION_NOT_WORK_REGION'               #
#  'REG_CITY_NOT_LIVE_CITY' 'REG_CITY_NOT_WORK_CITY'                        #
#  'LIVE_CITY_NOT_WORK_CITY' 'ORGANIZATION_TYPE' 'EXT_SOURCE_1'             #
#  'EXT_SOURCE_2' 'EXT_SOURCE_3' 'APARTMENTS_AVG' 'BASEMENTAREA_AVG'        #
#  'YEARS_BEGINEXPLUATATION_AVG' 'YEARS_BUILD_AVG' 'COMMONAREA_AVG'         #
#  'ELEVATORS_AVG' 'ENTRANCES_AVG' 'FLOORSMAX_AVG' 'FLOORSMIN_AVG'          #
#  'LANDAREA_AVG' 'LIVINGAPARTMENTS_AVG' 'LIVINGAREA_AVG'                   #
#  'NONLIVINGAPARTMENTS_AVG' 'NONLIVINGAREA_AVG' 'APARTMENTS_MODE'          #
#  'BASEMENTAREA_MODE' 'YEARS_BEGINEXPLUATATION_MODE' 'YEARS_BUILD_MODE'    #
#  'COMMONAREA_MODE' 'ELEVATORS_MODE' 'ENTRANCES_MODE' 'FLOORSMAX_MODE'     #
#  'FLOORSMIN_MODE' 'LANDAREA_MODE' 'LIVINGAPARTMENTS_MODE'                 #
#  'LIVINGAREA_MODE' 'NONLIVINGAPARTMENTS_MODE' 'NONLIVINGAREA_MODE'        #
#  'APARTMENTS_MEDI' 'BASEMENTAREA_MEDI' 'YEARS_BEGINEXPLUATATION_MEDI'     #
#  'YEARS_BUILD_MEDI' 'COMMONAREA_MEDI' 'ELEVATORS_MEDI' 'ENTRANCES_MEDI'   #
#  'FLOORSMAX_MEDI' 'FLOORSMIN_MEDI' 'LANDAREA_MEDI' 'LIVINGAPARTMENTS_MEDI'#
#  'LIVINGAREA_MEDI' 'NONLIVINGAPARTMENTS_MEDI' 'NONLIVINGAREA_MEDI'        #
#  'FONDKAPREMONT_MODE' 'HOUSETYPE_MODE' 'TOTALAREA_MODE'                   #
#  'WALLSMATERIAL_MODE' 'EMERGENCYSTATE_MODE' 'OBS_30_CNT_SOCIAL_CIRCLE'    #
#  'DEF_30_CNT_SOCIAL_CIRCLE' 'OBS_60_CNT_SOCIAL_CIRCLE'                    #
#  'DEF_60_CNT_SOCIAL_CIRCLE' 'DAYS_LAST_PHONE_CHANGE' 'FLAG_DOCUMENT_2'    #
#  'FLAG_DOCUMENT_3' 'FLAG_DOCUMENT_4' 'FLAG_DOCUMENT_5' 'FLAG_DOCUMENT_6'  #
#  'FLAG_DOCUMENT_7' 'FLAG_DOCUMENT_8' 'FLAG_DOCUMENT_9' 'FLAG_DOCUMENT_10' #
#  'FLAG_DOCUMENT_11' 'FLAG_DOCUMENT_12' 'FLAG_DOCUMENT_13'                 #
#  'FLAG_DOCUMENT_14' 'FLAG_DOCUMENT_15' 'FLAG_DOCUMENT_16'                 #
#  'FLAG_DOCUMENT_17' 'FLAG_DOCUMENT_18' 'FLAG_DOCUMENT_19'                 #
#  'FLAG_DOCUMENT_20' 'FLAG_DOCUMENT_21' 'AMT_REQ_CREDIT_BUREAU_HOUR'       #
#  'AMT_REQ_CREDIT_BUREAU_DAY' 'AMT_REQ_CREDIT_BUREAU_WEEK'                 #
#  'AMT_REQ_CREDIT_BUREAU_MON' 'AMT_REQ_CREDIT_BUREAU_QRT'                  #
#  'AMT_REQ_CREDIT_BUREAU_YEAR'                                             #
#                                                                           #
#############################################################################



# del train['TARGET']

'''
    FUNCTION DEFINITIONS
'''

def matrix_dist(A,B):
    A_q,A_r = np.linalg.qr(A)
    B_q,B_r = np.linalg.qr(B)
    return np.trace( ( A_q*B_r ) - (A_r*B_q  ) )


def mi(x, y, bins):
    # smooth for zero entries
    c_xy = np.histogram2d(x, y, bins)[0]+1
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    return 0.5 * g / c_xy.sum()

# vectorize function?
# vectorized_mi = np.vectorize(mi)

def clean(df):
    str_cols = df.select_dtypes(include='object').columns.values
    for col in str_cols:
        unique = df[col].unique()
        labels = np.arange(len(unique))
        df[col]=df[col].replace( dict( zip(unique,labels ) ))
    return df

def mi_matrix(data_set,create_fig=True):

    df = data[data_set]  

    # normalize strings
    str_cols = df.select_dtypes(include='object').columns.values
    for col in str_cols:
        unique = df[col].unique()
        labels = np.arange(len(unique))
        df[col]=df[col].replace( dict( zip(unique,labels ) ))

    # get the number of columns and rows
    # create empty matrix
    (n_rows,n_cols) = df.shape
    mi_matrix = np.zeros((n_cols,n_cols))

    data_matrix = df.values
    del df
    # Create list of indeces to reduce compute time
    jIDCS = list(range(1,n_cols))
    for i in range(n_cols-1):
        for j in jIDCS:
            mi_matrix[i,j] = mi(data_matrix[:,i],data_matrix[:,j],20 )
    if create_fig:
        fig = figure()
        fig.suptitle('Dataset: {}'.format(data_set), fontsize=20)
        imshow(mi_matrix, cmap='hot', interpolation='nearest')
    return mi_matrix


'''
    VARIABLE DEFINITIONS
'''

train = pd.read_csv('../data/application_train.csv').fillna(.0)
test = pd.read_csv('../data/application_test.csv').fillna(.0)

data = {'train':clean(train),'test':clean(test)}



def main():
    import json

    for ds in ['train','test']:
        df = data[ds]
        pca = PCA(n_components=len(df.T))
        scaled = pd.DataFrame(preprocessing.scale(df),columns = df.columns) 
        df = pca.fit_transform(scaled)
        print(df.shape)
        # table = pd.DataFrame(pca.components_,columns=train.columns,index = ['PC-{}'.format(i+1) for i in range(len(train.T))])
        fig = figure()
        fig.suptitle('Principle Components for {}'.format(ds))
        plot(pca.explained_variance_)
    targets = train['TARGET']
    del train['TARGET']
    mi = mutual_info_classif(train,targets)
    table = dict(zip(train.columns.values,mi))
    with open('features.json','w') as f:
        json.dump(table_sorted,f)
    with open('features.sorted.txt') as f:
        for k in sorted(table, key=table.get, reverse=True):
            f.write('{} {}'.format(k,table[k]))
    # fig = figure()
    # fig.suptitle('Mutual Information of Training')
    # bar(np.arange(len(mi)),mi)
    show()


if __name__ == '__main__':
    main()