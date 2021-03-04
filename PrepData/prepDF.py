# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 14:28:24 2021

@author: karan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:43:43 2021

@author: karan
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,mean_squared_error
import scipy
import pickle
import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
ps=PorterStemmer()
lm=WordNetLemmatizer()
nltk.download('stopwords')
s_words=set(stopwords.words('english'))

#Class to implement the preprocessing pipeline

###Use the trainPipeline() and testPipeline() functions to automate 


# Remove missing values from the dataset
# Args: na_method- method with which missing values are dealt with
#                  possible values- drop(default), mode, mean
def missingVals(df,na_method ='drop'):
    null=pd.DataFrame(df.isnull().sum())

    if null.sum()[0]==0:
        return df
    else:
        if na_method=='drop':
            df=df.dropna()
        elif na_method=='mode':
            for col in df.columns:
                df[col]=df[col].fillna(value=df[col].mode()[0])
        elif na_method=='mean':
            for col in df.columns:
                if df[col].dtypes=='O' or df[col].dtypes=='object': 
                    df[col]=df[col].fillna(value=df[col].mode()[0])
                else:
                    df[col]=df[col].fillna(value=df[col].mean())
        else:
            raise Exception('Invalid value for argument na_method')
        return df

    
# Converting categorical columsn to numerical 
# Args: ohe - True if columns to be one hot encoded, False for label encoding
#       dropFirst - if ohe is True, indicates if first column for each ohe conversion is to be dropped
def catEncoding(df,ohe=True,dropFirst=False):
    cat_col=[]
    for col in df.columns:
        if df[col].dtypes=='object':
            cat_col.append(col)
    
    if (len(cat_col)==0):
        return df
    if ohe==True and dropFirst==True:
        df=pd.get_dummies(df,columns=cat_col,drop_first=True)
    elif ohe==True and dropFirst==False:
        df=pd.get_dummies(df,columns=cat_col,drop_first=False)
    else:
        le=LabelEncoder()
        for col in cat_col:
            df[col]=le.fit_transform(df[col])
    return df


# Remove outliers from the dataset
# Args: n_std- specifies the number of standard deviations upto which the values are to be kept 
#              default = 3
def remOutliers(df,n_std=3):
    
    z_scores = scipy.stats.zscore(df)        
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < n_std).all(axis=1)
    new_df = df[filtered_entries]
    return new_df


# Scale the values in the dataset
# Args: scale_list- list of all columns to be scaled. default: all columns
#       type- type of scaling. 'std' for standard scaling and 'minmax' for min-max scaling
def scaleVals(df,scale_list=None,scale_type='std'):
    
    if scale_list==None:
        scale_list=list(df.columns)

    if scale_type=='minmax':
        mm=MinMaxScaler()
        df[scale_list]=mm.fit_transform(df[scale_list])
    elif scale_type=='std':
        sc=StandardScaler()
        df[scale_list]=sc.fit_transform(df[scale_list])
    else:
        raise Exception('Invalid value for argument scale_type)')
    return df


#Function to split dataset into test and train
def testSplit(X,y,test_split=0.2):
    if type(X)==pd.core.frame.DataFrame:
        X=X.values
    if type(y)==pd.core.frame.DataFrame:
        y=y.values
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_split)
    return X_train,X_test,y_train,y_test


# Train the random forest model on the preprocessed model and save the model as a pickle file
# Returns: the accuracy of model and the model
# Args: X- set of all features to be used for prediction
#       y - Target variable
#       test_split - the ratio of the test set
#       folds - number of folds for k-fold cross val
#       model_name- name of the model pkl file to be saved  
#       task_type- 'c' for classification and 'r' for regression
def splitAndTrain(X,y,test_split=0.2,folds=5,task_type='c',model_name='model'):
    
    if type(X)==pd.core.frame.DataFrame:
        X=X.values
    if type(y)==pd.core.frame.DataFrame:
        y=y.values
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_split)
    if task_type=='c':
        rf=RandomForestClassifier(n_estimators=10)   
        rf.fit(X_train,y_train)
        with open(model_name+'.pkl', 'wb') as file:
            pickle.dump(rf, file)
        score=cross_val_score(rf,X,y,cv=folds)
        return np.mean(score),rf
    elif task_type=='r':
        lr=LinearRegression()
        lr.fit(X_train,y_train)
        with open(model_name+'.pkl', 'wb') as file:
            pickle.dump(lr, file)
        score=cross_val_score(lr,X,y,cv=folds,scoring='r2')
        return np.mean(score),lr
    else:
        raise Exception('Invalid value for argument task_type')
        


def prepText(df,col,na_method='drop',stopword=True,lemmatize=True,lem_method='l'):
    
    df=df.reset_index(drop=True)
    if na_method=='drop':
        null=[]
        for i in range(len(df[col])):
            if str(df[col][i])=='nan':
                null.append(i)
            
        new_df=df.drop(null)
    elif type(na_method)==str:
        for i in range(len(df[col])):
            if str(df[col][i])=='nan':
                df[col][i]=na_method
    
    else:
        raise Exception('argument na_method must be of type string')

    df=df.reset_index(drop=True)    
    corpus=[]
    for i in range(len(df)):
        
        temp=re.sub('[^a-zA-Z0-9]',' ',df[col][i])
        temp=temp.lower()
        temp=temp.split()
        
        if stopword==True:
            temp=[word for word in temp if word not in s_words]
        if lemmatize==True:
            if lem_method=='l':
                temp=[lm.lemmatize(word) for word in temp]
            elif lem_method=='s':
                temp=[ps.stem(word) for word in temp]
            else:
                raise Exception('Invalid value for argument lem_method')
        temp=[word for word in temp if len(word)>2]
        temp=' '.join(temp)
        
        corpus.append(temp)
        
    return corpus


# Main function to run the complete data preprocessing and model training pipeline
# Args: dataframe - input dataframe
#       features - list of columns in the dataframe to be used as features
#       target - name of target column
#       rem_outliers- whether outliers need to be removed (default: True)
#       scale_vals- whether values need to be scaled (default: True)
# All the remaining arguments are same as used in the above functions
# 
#Returns: model - the model trained on the given data
#         X - preprocessed dataframe
#         y - corresponding target variable after preprocessing
def trainPipeline(dataframe,features,target,na_method='drop',ohe=True,
                  dropFirst=False,rem_outliers=True,n_std=3,scale_vals=True,
                  scale_list=None,scale_type='std',test_split=0.2,folds=5,task_type='c',model_name='model'):
    s=locals()
    with open('saved_args.pkl', 'wb') as file:
        pickle.dump(s, file)
    features_temp=features.copy()
    features_temp.append(target)
    df=dataframe[features_temp]
    df=missingVals(df,na_method)
    df=catEncoding(df,ohe,dropFirst)
    if rem_outliers==True:
        df=remOutliers(df,n_std)
    if scale_vals==True:
        if scale_list==None:
            scale_list2=list(df.columns)
            scale_list2.remove(target)
            df=scaleVals(df,scale_list2,scale_type)
        else:
            df=scaleVals(df,scale_list,scale_type)
    
    y=df[target]
    X=df.drop(target,axis=1)

    acc,model=splitAndTrain(X.values,y.values,test_split,folds,task_type,model_name)
    print(f'Accuracy with {folds} folds = {acc*100}%')
    return model, X, y



# Main function to run the complete prediction pipeline
# Args: dataframe - input dataframe
#       features - list of columns in the dataframe to be used as features
#       model_name - name of the model saved in the trainPipeline()
# Remaining arguments are the same as trainPipeline()
# Returns: pred - array of predictions
def predictPipeline(dataframe,features,na_method='drop',ohe=True,
                  dropFirst=False,rem_outliers=True,n_std=3,scale_vals=True,
                  scale_list=None,scale_type='std',model_name='model'):
    
    
    args=locals()
    args.pop('dataframe')
    
    
    #Checking that the trainPipeline() should be run before predictPipeline()
    if os.path.exists('saved_args.pkl')==0:
        raise Exception('Must run trainPipeline() before predictPipeline()')
    #Checking for identical predict and train parameters
    with open('saved_args.pkl','rb') as file:           
        saved_args=pickle.load(file)
    rem_list=['dataframe','folds','target','test_split','task_type']
    [saved_args.pop(key) for key in rem_list]

    if args!=saved_args:
        raise Exception('Test arguments must be same as Train arguments')
    
    df=dataframe[features]
    df=missingVals(df,na_method)
    df=catEncoding(df,ohe,dropFirst)
    if rem_outliers==True:
        df=remOutliers(df,n_std)
    if scale_vals==True:
        df=scaleVals(df,scale_list,scale_type)
    
    with open(model_name+'.pkl','rb') as file:
        model=pickle.load(file)
    
    pred=model.predict(df.values)
    return pred



# Main function to process the dataframe
# Args: dataframe - input dataframe
#       features - list of columns in the dataframe to be used as features
#       
# Remaining arguments are the same as trainPipeline()
# Returns: X - the processed features dataframe
#          y - pandas series containing the target variable
def processDf(dataframe,features,target,na_method='drop',ohe=True,
                  dropFirst=False,rem_outliers=True,n_std=3,scale_vals=True,
                  scale_list=None,scale_type='std'):
    features_temp=features.copy()
    features_temp.append(target)
    df=dataframe[features_temp]
    df=missingVals(df,na_method)
    df=catEncoding(df,ohe,dropFirst)
    if rem_outliers==True:
        df=remOutliers(df,n_std)
    if scale_vals==True:
        if scale_list==None:
            scale_list2=list(df.columns)
            scale_list2.remove(target)
            df=scaleVals(df,scale_list2,scale_type)
        else:
            df=scaleVals(df,scale_list,scale_type)
    y=df[target]
    X=df.drop(target,axis=1)
    
    return X,y


# Function to process the dataframe and split into test and train set
def processAndSplit(dataframe,features,target,na_method='drop',ohe=True,
                  dropFirst=False,rem_outliers=True,n_std=3,scale_vals=True,
                  scale_list=None,scale_type='std',test_split=0.2):
    features_temp=features.copy()
    features_temp.append(target)
    df=dataframe[features_temp]
    df=missingVals(df,na_method)
    df=catEncoding(df,ohe,dropFirst)
    if rem_outliers==True:
        df=remOutliers(df,n_std)
    if scale_vals==True:
        if scale_list==None:
            scale_list2=list(df.columns)
            scale_list2.remove(target)
            df=scaleVals(df,scale_list2,scale_type)
        else:
            df=scaleVals(df,scale_list,scale_type)
    
    y=df[target]    
    X=df.drop(target,axis=1)
    return testSplit(X,y)


# Score the models predictions
#Args: y_true- the actual labels
#      y_pred- predictions made by the model
#      type- the type of task. 'c' for classification (default) and 'r' for regression
def predScore(y_true,y_pred,task_type='c'):
    
    if task_type=='c':
        print('Accuracy=',accuracy_score(y_true,y_pred))
    elif task_type=='r':
        print('MSE score=',mean_squared_error(y_true,y_pred))
    else:
        raise Exception("Invalid 'type' in predScore()")
