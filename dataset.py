import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


def read_dataset(test_size,data_set_type,relevant_columns,lable_column_name):
  

  # Read Train data
  df=None

  if(data_set_type == 'kaggle'):
    df = pd.read_csv('dataset/train-kaggle.csv')  
  else:
    df = pd.read_csv('dataset/train-google.csv')  

  
  # amend train data
  if(data_set_type == 'kaggle'):
    df['Cabin'].replace(np.NaN,'',regex=True,inplace=True)
  df['Embarked'].replace(np.NaN,'',regex=True,inplace=True)
  df['Age'].replace(np.NaN,-1,inplace=True)

  for col in df.columns:
    if(not col in relevant_columns):
      df.pop(col)

  df_train,df_test = train_test_split(df,test_size=test_size,shuffle=True)
  y_train = df_train.pop(lable_column_name)
  y_test = df_test.pop(lable_column_name)

  # Read Test Data
  df_predict = None
  y_predict = None
  if(data_set_type == 'kaggle'):
    df_predict = pd.read_csv('dataset/predict-kaggle.csv')
  else:
    df_predict = pd.read_csv('dataset/predict-google.csv')
    y_predict = df_predict.pop(lable_column_name)
  
  # amend predict data
  if(data_set_type == 'kaggle'):
    df_predict['Cabin'].replace(np.NaN,'',regex=True,inplace=True)
  df_predict['Embarked'].replace(np.NaN,'',regex=True,inplace=True)
  df_predict['Age'].replace(np.NaN,-1,inplace=True)

  for col in df_predict.columns:
    if(not col in relevant_columns):
      df_predict.pop(col)

  return df_train,y_train,df_test,y_test,df_predict,y_predict