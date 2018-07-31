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
train = pd.read_csv('application_train.csv')

# Initial inspection ----------------------------------------------------------
credit.columns
credit.head(10)
credit.shape
credit.describe()

def missing(df):
    """Percent missing values by column."""
    cols = df.columns[df.isnull().any()].tolist()
    result = df[cols].isnull().sum() / df.shape[0] * 100
    result.to_frame().reset_index()
    return result

missing(credit)
# Only 3 unique percentages --> suggests that columns often missing in 
# combination with one another (i.e. if missing one, likely to be missing 
# the others)

# Specific topics from issue 8 ------------------------------------------------
# (1) Number/percentage of applicants with previous credits in this CSV
total_applicants = len(train['SK_ID_CURR'].unique()) 
credit_applicants = len(credit['SK_ID_CURR'].unique()) 
percent = credit_applicants / total_applicants * 100
print('There are {} unique applicants in credit_card_balance.csv \nThis is {:.2f}% of the total applicants in applications_train.csv'.
      format(credit_applicants, percent))

# (2) Avg number of credits in the table per applicant with at least one credit
counts = credit.groupby('SK_ID_CURR').count()[['MONTHS_BALANCE']]
before = len(counts)
counts.query('MONTHS_BALANCE > 1', inplace=True)
avg_len = counts.values.mean()
print('{:.2f}% of applicants have > 1 month credit balance \nFor these applicants, the average length of credit is {:.1f} months'.
      format(len(counts) / before * 100, avg_len))

counts.hist(bins=30) # use more bins to better visualize distribution
# The distribution of months balance is bimodal with a slight skew to the left.
# I would have expected it to be more exponential, so the higher prevalence of 
# long-term loans (> 75 months) compared to mid-term loans (50-75 months) is 
# unexpected. Maybe this is because around 50 months is when applicants decide
# whether they want to keep using their current credit card or switch to a 
# different one.

# (3) Understand features: systemic missing-ness, representation, processing