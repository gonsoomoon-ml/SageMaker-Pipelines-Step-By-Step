import argparse
import os
import requests
import tempfile
import subprocess, sys


# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy==1.19.5'])

import numpy as np
print("numpy version: ", np.__version__)



import pandas as pd

from glob import glob

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import logging
import logging.handlers

def _get_logger():
    '''
    로깅을 위해 파이썬 로거를 사용
    # https://stackoverflow.com/questions/17745914/python-logging-module-is-printing-lines-multiple-times
    '''
    loglevel = logging.DEBUG
    l = logging.getLogger(__name__)
    if not l.hasHandlers():
        l.setLevel(loglevel)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))        
        l.handler_set = True
    return l  

logger = _get_logger()

def merge_X_y(X_pre, y_pre):
    '''
    '''
    X = np.concatenate((y_pre, X_pre), axis=1)
    train, validation, test = np.split(X, [int(.7*len(X)), int(.85*len(X))])
    
    train_df = pd.DataFrame(train)
    train_df[[0]] = train_df[[0]].astype('int') # converto a type of label_column to int
    
    val_df = pd.DataFrame(validation)
    val_df[[0]] = val_df[[0]].astype('int') # converto a type of label_column to int

    test_df = pd.DataFrame(test)
    test_df[[0]] = test_df[[0]].astype('int') # converto a type of label_column to int

    return train_df, val_df, test_df
    

if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_output_dir', type=str, default="/opt/ml/processing/output")
    parser.add_argument('--base_preproc_input_dir', type=str, default="/opt/ml/processing/input")   
    parser.add_argument('--label_column', type=str, default="fraud")       
    # parse arguments
    args = parser.parse_args()     
    
    logger.info("#############################################")
    logger.info(f"args.base_output_dir: {args.base_output_dir}")
    logger.info(f"args.base_preproc_input_dir: {args.base_preproc_input_dir}")    
    logger.info(f"args.label_column: {args.label_column}")        
    
    ##############################################

    base_output_dir = args.base_output_dir
    base_preproc_input_dir = args.base_preproc_input_dir
    label_column = args.label_column    

    input_files = glob('{}/*.csv'.format(base_preproc_input_dir))
    logger.info(f"input files: \n {input_files}")    
    
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(base_preproc_input_dir, "train"))
        



    raw_data = [ pd.read_csv(file) for file in input_files ]
    df = pd.concat(raw_data)
   
    logger.info(f"dataframe shape \n {df.shape}")    
    logger.info(f"dataset sample \n {df.head(2)}")    
    
    logger.info(f"df columns \n {df.columns}")    
    
    # Extract label_column from dataframe. After that, the dataframe doesn't have label columns
    y = df.pop(label_column)

        
    float_cols = df.select_dtypes(include=['float64']).columns.values
    int_cols = df.select_dtypes(include=['int64']).columns.values
    numeric_features = np.concatenate((float_cols, int_cols), axis=0).tolist()
    
#     numeric_features = list(feature_columns_names)
#     numeric_features.remove("sex")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_features = df.select_dtypes(include=['object']).columns.values.tolist()    
#     categorical_features = ["sex"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    
    
    X_pre = preprocess.fit_transform(df)
    y_pre = y.astype('int').to_numpy().reshape(len(y), 1)
    
    train, validation, test = merge_X_y(X_pre, y_pre)   
#     print("y_pre type: ", type(y_pre))
#     print("y_pre: ", y_pre)
#    X = np.concatenate((y_pre, X_pre), axis=1)
#    print("X: ", X)

    
    # np.random.shuffle(X)
 #   train, validation, test = np.split(X, [int(.7*len(X)), int(.85*len(X))])
    
#     train_df = pd.DataFrame(train)
#     train_df[[0]] = train_df[[0]].astype('int')
#     print(train_df.head())

    train.to_csv(f"{base_output_dir}/train/train.csv", header=False, index=False)
    validation.to_csv(f"{base_output_dir}/validation/validation.csv", header=False, index=False)
    test.to_csv(f"{base_output_dir}/test/test.csv", header=False, index=False)
    
    print(train.head(2))
#     pd.DataFrame(train).to_csv(f"{base_output_dir}/train/train.csv", header=False, index=False)
#     pd.DataFrame(validation).to_csv(f"{base_output_dir}/validation/validation.csv", header=False, index=False)
#     pd.DataFrame(test).to_csv(f"{base_output_dir}/test/test.csv", header=False, index=False)
    
    logger.info(f"preprocessed train shape \n {train.shape}")        
    logger.info(f"preprocessed validation shape \n {validation.shape}")            
    logger.info(f"preprocessed test shape \n {test.shape}")            
    
    logger.info(f"preprocessed train path \n {base_output_dir}/train/train.csv")
    logger.info(f"preprocessed train sample \n {pd.DataFrame(train).head(2)}")
    logger.info(f"All files are preprocessed")
