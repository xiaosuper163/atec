## fill nas with median
## very rude transformation of dates
## test/train split
import pandas as pd
import numpy as np
data = pd.read_csv("atec_anti_fraud_train.csv",parse_dates=['date'])

from pandas.api.types import is_numeric_dtype

def fix_missing(train, test, col):
    if is_numeric_dtype(train[col]):
        if train[col].isnull().sum():
            train[col+'_na'] = pd.isnull(train[col])
            test[col+'_na'] = pd.isnull(test[col])
            filler = train[col].median()
            train[col] = train[col].fillna(filler)
            test[col] = test[col].fillna(filler)
    return train, test

test = pd.read_csv("atec_anti_fraud_test_a.csv",parse_dates=['date'])

for col in data.columns:
    data, test = fix_missing(data, test, col)

def process_dates(df,date):
    attrs = ['Day', 'Dayofweek', 'Is_month_end', 'Is_month_start']
    for attr in attrs:
        df[attr] = getattr(df[date].dt, attr.lower())
    return df

data = data[data['label']!=-1]

data.sort_values('date',inplace=True)

data = process_dates(data,'date')
test = process_dates(test,'date')

train = data.iloc[:len(data) * 4 // 5]
valid = data.iloc[len(data)*4//5:]

train.drop('date',axis = 1, inplace=True)
valid.drop('date',axis=1,inplace=True)

features = list(train.columns)[2:]

train_x, train_y = train[features],train['label']
valid_x,valid_y = valid[features],valid['label']