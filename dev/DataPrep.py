import os
import numpy as np
import torch
import json
import sklearn as sk
import pandas as pd
from sklearn.preprocessing import Imputer,scale
from scipy.stats import chi2_contingency
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

# IMPORT THESE FUNCTIONs
def clean(df):
    str_cols = df.select_dtypes(include='object').columns.values
    for col in str_cols:
        unique = df[col].unique()
        labels = np.arange(len(unique))
        df[col]=df[col].replace( dict( zip(unique,labels ) ))
    return df

def calc_n_feats(fname):
    '''
        perfom PCA on dataset
        then calculate where the curve on the explained variance with the largest distance between points

        NOTE: using aic or bic might work better
            - https://en.wikipedia.org/wiki/Bayesian_information_criterion
            - https://en.wikipedia.org/wiki/Akaike_information_criterion
            - https://stats.stackexchange.com/questions/577/is-there-any-reason-to-prefer-the-aic-or-bic-over-the-other

        maybe even MDL - https://en.wikipedia.org/wiki/Minimum_description_length

        I don't know
    '''
    train = clean(pd.read_csv(fname).fillna(.0)) 
    targets = train['TARGET']
    # remove targets 
    del train['TARGET']  

    pca = PCA(n_components=len(train.T))
    scaled = pd.DataFrame(preprocessing.scale(train),columns = train.columns) 
    pca.fit_transform(scaled)

    arr = pca.explained_variance_ratio_
    
    nPoints = len(arr)
    
    allCoords = np.vstack((range(nPoints), arr)).T
    
    # enumerate vector
    np.array([range(nPoints), arr])
    firstPoint = allCoords[0]
    lineVec = allCoords[-1] - allCoords[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoords - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint

    
def get_feats(train,nfeats=None,to_disk=False):
    from itertools import islice         
    try:
        targets = train['TARGET']
        del train['TARGET']   
    except:
        raise ValueError('No "TARGET" column in dataset, THIS IS REQUIRED')

    if not nfeats:
        nfeats = calc_n_feats(train)

    mi = mutual_info_classif(train,targets)
    table = dict(zip(train.columns.values,mi))
    best = sorted(table, key=table.get, reverse=True)
    if to_disk:
        with open('features.json','w') as f:
            json.dump(table_sorted,f)
        with open('features.sorted.txt') as f:
            for k in best:
                f.write('{} {}'.format(k,table[k]))
    return islice(iterable, nfeats)

class DataSetProto(DataSet):
    '''
        Dataset Prototype:
        read a csv and convert to tensor
        if config is given, then calc n best features

        n: use pca
        feature columns: use MI (because I'm lazy)
    '''

    def __init__(self,fname,config=None, **kwargs):
        super(DataSetProt,self).__init__()

        for k,v in kwargs.items(): setattr(self,k,v)

        self.data=clean(pd.read_csv(train_fname).fillna(.0))
        features = None
        select_features = None
        if config:
            try:
                features = config['features']
                self.data = self.data[features]
            except:
                try:
                    if config['select_features']:
                        nfeats = list(get_feats(self.data).keys())
                        self.data = self.data[nfeats]
                except:
                    pass
                pass
        
        self.data = torch.tensor(self.data.values,dtype=torch.float32)
        

    def __getitem__(self, ix):
        return self.data[ix]

    def __len__(self):
        return len(self.data)

        


def prepare_data(args):
    
    '''
        Consume *.csv file path and reduce features
        and return a RiskDataset
    '''

    data_set = DataSetProto(args.data_file,args.config)
    
    # get "masks" in order to split the dataset into subsets
    num_train = len(data_set)
    indices = np.arange(num_train)
    other_mask = np.random.sample(np.sum(~mask)) < args.split
    train_ix = other_ix[other_mask]

    # use a sampler to specify which subsets correspond to training, validation, test
    data_sampler = SubsetRandomSampler(train_ix)

    kwargs = {'num_workers': 2} # specifies number of subprocesses to use for loading data
    if args.use_cuda:
        kwargs['pin_memory'] = True # speeds up cpu to gpu retrieval

    # need to include the samplers to make sure dataloaders pull from correct subsets
    data_loader = DataLoader(data_set, batch_size=args.batch_size,
        sampler=data_sampler, **kwargs)
    return data_loader
