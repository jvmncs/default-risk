#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 19:15:19 2018

@author: addisonklinke

EDA for credit_card_balance.csv dataset
Issue #8 on Github
"""

# Load modules and data -------------------------------------------------------
import pandas as pd
import os

os.chdir('./Documents/git/default-risk/data/')
credit = pd.read_csv('credit_card_balance.csv')
train = pd.read_csv('applicants_train.csv')

# Initial inspection ----------------------------------------------------------
credit.shape
credit.describe()

# Specific topics from issue 8 ------------------------------------------------
# (1) Number/percentage of applicants with previous credits in this CSV
total_applicants = ... # get this number from the application_train.csv table
print(len(credit.loc[:, 'applicant_id'].unique())/total_applicants) 

# (2) Avg number of credits in the table per applicant with at least one credit
counts = credit.group_by(applicant_id).count()
print(counts.iloc[:, -1].mean())

# (3) Understand features - systemic missing-ness, representation, processing