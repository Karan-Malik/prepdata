# PrepData
[![pypiversion](https://img.shields.io/pypi/v/prepdata)](https://pypi.org/project/prepdata/)
[![issues](https://img.shields.io/github/issues/Karan-Malik/prepdata)](https://github.com/Karan-Malik/prepdata/issues)
[![forks](https://img.shields.io/github/forks/Karan-Malik/prepdata)](https://github.com/Karan-Malik/prepdata/network/members)
[![stars](https://img.shields.io/github/stars/Karan-Malik/prepdata)](https://github.com/Karan-Malik/prepdata/stargazers)

*Automate Data Preprocessing for Data Science Projects*

Glide through the most repetitive part of Data Science, preprocessing the dataframes with [prepdata](https://pypi.org/project/prepdata/). This library lets you train your Machine Learning models without worrying about the imperfections of the underlying dataset. 

## Features
Use a single comprehensive function to automatically:
- Remove missing values
- Remove outliers
- Encode categorical variables
- Scale values
- Split the dataset
- Train your model  
- Process text data
## Table Of Contents

* [Sample Code](#sample-code)
* [Documentation](#documentation)
  + [Main Functions](#main-functions)
    - [trainPipeline()](#trainpipeline)
    - [predictPipeline()](#predictpipeline)
    - [processDf()](#processdf)
    - [processAndSplit()](#processandsplit)
    - [prepText()](#preptext)
  + [Sub Functions](#sub-functions)
    - [missingVals()](#missingvals)
    - [catEncoding()](#catencoding)
    - [remOutliers()](#remoutliers)
    - [scaleVals()](#scalevals)
    - [testSplit()](#testsplit)
    - [splitAndTrain()](#splitandtrain)
    - [predScore()](#predscore)


#### Install the library from Pypi [here](https://pypi.org/project/PrepData/)

## Sample Code

```Python
from PrepData import prepDF


#importing data
dataframe=pd.read_csv('Churn_Modelling.csv')

#defining features and target column
target_column='Exited'
features=list(set(dataframe.columns) - set(['RowNumber','CustomerId','Surname','Exited']))

#list of columns to be scaled
scale_list=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts','EstimatedSalary']

#using the main library functions
#the subfunctions documented below are to be used similarly

# 1) trainPipeline() 
model,X,Y=prepDF.trainPipeline(dataframe,features,target_column,dropFirst=True,scale_list=scale_list)

# 2) predictPipeline()
pred=prepDF.predictPipeline(dataframe, features,dropFirst=True,scale_list=scale_list)

# 3) processDF()
a,b=prepDF.processDf(dataframe,features,target_column,dropFirst=True,scale_list=scale_list)

# 4) processAndSplit()
x_tr,x_te,y_tr,y_te=prepDF.processAndSplit(dataframe, features, target_column,scale_list=scale_list,dropFirst=True)
```
## Documentation

The library works on [Pandas](https://pandas.pydata.org/) dataframes. All the available functions have been documented below.

### Main Functions
These are the main function which with the help of the sub functions process the data and help ease the tasks

#### trainPipeline()

  ```python
  def trainPipeline(dataframe,features,target,na_method='drop',ohe=True,
                        dropFirst=False,rem_outliers=True,n_std=3,scale_vals=True,
                        scale_list=None,scale_type='std',test_split=0.2,folds=5,model_name='model'):
    """
    Main function to run the complete data preprocessing and model training pipeline. Automatically replace missing values, encode categorical features, remove outliers, scale values, split the dataset and train the model.

    Parameters: 
    dataframe - the pandas dataframe to be used for processing (pandas.DataFrame)
    features - list of columns in the dataframe to be used as features (list)
    target - name of target column (string)
    na_method- method used to deal with the missing values (string).
      [ Possible values- 'drop' (default), 'mode' and 'mean' ]
    ohe - Method used for converting categorical columns (Boolean)
      [ Possible values- True for one hot encoding, False for label encoding ]
    dropFirst - Method to determine whether dummy variable is to be discarded or not. Valid only if ohe=True (Boolean).
      [ Possible values- True for dropping dummy variable, otherwise False ]
    rem_outliers -  whether outliers are to be removed (Boolean)
      [ Possible values- True for removing outliers (Default), otherwise False ]
    n_std- The number of standard deviations upto which the values are to be kept (int)
      [ Default - 3 ]
    scale_vals -  whether values are to be scaled or not (Boolean)
      [ Possible values- True for scaling (Default), False for no scaling ]
    scale_list- list of all columns to be scaled. Default: all columns (list).
    scale_type- Method used for scaling (string)
      [ Possible values- 'std' for standard scaling (default) and 'minmax' for min-max scaling ]
    test_split - Ratio of test set to the the total data (float). Default: 0.2
    folds - number of folds for k-fold cross validation (int). Default: 5
    task_type - type of task to be carried out by the random forest model (string).
      [ Possible values- 'c' for classification (default) and 'r' for regression ]
    model_name - name of the model pkl file to be saved (string). Default: 'model'

    Returns: 
    model - sklearn model (Random Forest) trained on the given dataset.
    X - Preprocessed dataframe used to train model
    y - target vector
    """
  ```

#### predictPipeline()

  ```python
  def predictPipeline(dataframe,features,na_method='drop',ohe=True,
                        dropFirst=False,rem_outliers=True,n_std=3,scale_vals=True,
                        scale_list=None,scale_type='std',model_name='model'):
    """
    Main function to run the complete prediction pipeline using the model trained with the trainPipeline() function. The trainPipeline() function must be executed before using the predictPipeline().

    Arguments for predictPipeline() must be identical to trainPipeline() to ensure that the processed dataframes are identical in both cases.  

    Parameters: 
    dataframe - the pandas dataframe to be used for predictions (pandas.DataFrame)
    features - list of columns in the dataframe to be used as features (list)
    model_name - name of the model saved in the trainPipeline() (string).
    ohe - Method used for converting categorical columns (Boolean)
      [ Possible values- True for one hot encoding, False for label encoding ]
    dropFirst - Method to determine whether dummy variable is to be discarded or not. Valid only if ohe=True (Boolean).
      [ Possible values- True for dropping dummy variable, otherwise False ]
    rem_outliers -  whether outliers are to be removed (Boolean)
      [ Possible values- True for removing outliers (Default), otherwise False ]
    n_std- The number of standard deviations upto which the values are to be kept (int)
      [ Default - 3 ]
    scale_vals -  whether values are to be scaled or not (Boolean)
      [ Possible values- True for scaling (Default), False for no scaling ]
    scale_list- list of all columns to be scaled. Default: all columns (list).
    scale_type- Method used for scaling (string)
      [ Possible values- 'std' for standard scaling (default) and 'minmax' for min-max scaling ]

    Returns: 
    pred - array of predictions made using the given dataframe and model.
    """
  ```

#### processDf()

  ```python
  def processDf(dataframe,features,target,na_method='drop',ohe=True,
                    dropFirst=False,rem_outliers=True,n_std=3,scale_vals=True,
                    scale_list=None,scale_type='std'):
    """
    Function to preprocess the dataframe. Similar to trainPipeline() except no model is trained and returned. 

    Parameters: 
    Arguments are identical to trainPipeline().

    Returns: 
    X - Preprocessed dataframe for model training
    y - target vector
    """
  ```

#### processAndSplit()

  ```python
  def processAndSplit(dataframe,features,target,na_method='drop',ohe=True,
                    dropFirst=False,rem_outliers=True,n_std=3,scale_vals=True,
                    scale_list=None,scale_type='std',test_split=0.2):
    """
    Function to preprocess the dataframe and split into train and test set. Similar to processDF() except the processed dataframe is split and returned. 

    Parameters: 
    Arguments are identical to trainPipeline().

    Returns: 
    4 Dataframes -  X_train,X_test,y_train,y_test
    """
  ```

#### prepText()

  ```python
  def prepText(df,col,na_method='drop',stopword=True,lemmatize=True,lem_method='l'):
  
  """
    Main function to preprocess text data.

    Parameters: 
    df - the pandas dataframe to be used (pandas.DataFrame)
    col - column in the dataframe containing the text (string)
    na_method- method used to deal with the missing values (string).
      [ Possible values- 'drop' (default) or any other string to replace the missing value ]
    stopword- whether stopwords are to be removed or not (Boolean).
      [ Possible values- True for one hot encoding, False for label encoding ]
    lemmatize- whether words are to be lemmatized/stemmed (Boolean).
      [ Possible values- True (default), False ]
    lm_method- method to be used for lemmatizing/stemming (string).
      [ Possible values- 'l' for lemmatizing, 's' for stemming ]
    
    Returns: 
    corpus - list containing the preprocessed text.
    """
  ```

### Sub Functions
These are the functions used inside the main functions

#### missingVals()

  ```python
  def missingVals(df,na_method ='drop'):
    """
    Remove or replace missing values from the dataset

    Parameters:
    df - dataframe to be used (pandas.DataFrame).
    na_method - method used to deal with the missing values (string). 
      [ Possible values- 'drop' (default), 'mode' and 'mean']

    Returns:
    pandas.DataFrame :Dataframe without missing values
    """
  ```

#### catEncoding()

  ```python
  def catEncoding(df,ohe=True,dropFirst=False):
    """
    Converting categorical columns to numerical, using label encoding or one-hot encoding.
    
    Parameters: 
    df - dataframe to be used (pandas.DataFrame).
    ohe - Method used for converting categorical columns (Boolean)
     [ Possible values - True for one hot encoding, False for label encoding ]
    dropFirst - Method to determine whether dummy variable is to be discarded or not. Valid only if ohe=True (Boolean).
     [ Possible values - True for dropping dummy variable, otherwise False ]

    Returns: 
    pandas.dataframe :Dataframe with all numerical columns
    """
  ```

#### remOutliers()

  ```python
  def remOutliers(df,n_std=3):
    """
    Remove outliers from the dataset using number of standard deviations (z-score).

    Parameters: 
    df - dataframe to be used (pandas.DataFrame).
    n_std - The number of standard deviations upto which the values are to be kept (float).
      [ Default - 3.0 ]

    Returns: 
    pandas.dataframe :Dataframe without outliers.
    """
  ```

#### scaleVals()

  ```python
  def scaleVals(df,scale_list=None,scale_type='std'):
    """
    Scale the values present in the dataset

    Parameters: 
    df - dataframe to be used (pandas.DataFrame).
    scale_list - list of all columns to be scaled. Default: all columns (list).
    scale_type - Method used for scaling (string)
      [ Possible values - 'std' for standard scaling (default) and 'minmax' for min-max scaling ]

    Returns: 
    Dataframe with scaled values
    """
  ```

#### testSplit()

  ```python
  def testSplit(X,y,test_split=0.2):
    """
    Split the dataset into training and test set

    Parameters: 
    X - dataframe with all the features (pandas.DataFrame/ numpy.array).
    y - target column (pandas.DataFrame/ numpy.array). 
    test_split - Ratio of test set to the the total data (float). Default: 0.2
                     
    Returns: 
    4 Dataframes -  X_train,X_test,y_train,y_test
    """
  ```

#### splitAndTrain()

  ```python
  def splitAndTrain(X,y,test_split=0.2,folds=5,task_type='c',model_name='model'):
    """
    Split the dataset into train and test test, train a random forest model on the training set and save the model as a pickle file.

    Parameters: 
    X - dataframe with all the features (pandas.DataFrame/ numpy.array).
    y - target column (pandas.DataFrame/ numpy.array).
    scale_list - list of all columns to be scaled (list). Default: all columns.
    test_split - Ratio of test set to the the total data (float). Default: 0.2
    folds - number of folds for k-fold cross validation (int). Default: 5
    task_type - type of task to be carried out by the random forest model (string).
      [ Possible values- 'c' for classification (default) and 'r' for regression ]
    model_name - name of the model pkl file to be saved (string). Default: 'model'

    Returns: 
    score - accuracy of the model trained. Accuracy for classification and R2 score for regression.
    model - sklearn model (Random Forest) trained by the function on the given dataset.
    """
  ```

#### predScore()

  ```python
  def predScore(y_true,y_pred,task_type='c'):
    """
    Score the models predictions

    Parameters: 
    y_true- Vector of actual labels from the test set (numpy.array)
    y_pred- Vector of predictions made by the model (numpy.array).
    task_type - type of task that was carried out (string).
      [ Possible values- 'c' for classification (default) and 'r' for regression ]
                     
    Prints:  
    Prints the accuracy value for classification and MSE score for regression.

    Returns: 
    None
    """
  ```

#### In case of any suggestions or problems, feel free to open an [issue](https://github.com/Karan-Malik/prepdata/issues)
---------------------------------------------------------------------------------------

*Copyright (c) 2021 Karan-Malik*
















