# Automate Data Preprocessing for Data Science Projects
Glide through the most repetitive part of Data Science, preprocessing the dataframes with [prepdata](https://github.com/Karan-Malik/prepdata). prepdata lets you train your Machine Learning models without worrying about the imperfections of the underlying dataset. Replace missing values, remove outliers, encode categorical variables, split the dataset and train your model automatically with the comprehensive functions of this library. 

### https://pypi.org/project/prepdata/0.1.0

## Usage
The library works on [Pandas](https://pandas.pydata.org/) dataframes. All the available functions have been documented below.

### Sub-functions:

#### 1) **missingVals()** - Remove or replace missing values from the dataset


```Python
def missingVals(df,na_method ='drop')
```

Arguments: <br><br>
*df*- dataframe to be used (pandas.DataFrame).

*na_method*- method used to deal with the missing values (string).
                 
       Possible values- 'drop' (default), 'mode' and 'mean'

Returns: <br><br>*Dataframe without missing values.*

<br>


#### 2) **catEncoding()** - Converting categorical columns to numerical, using label encoding or one-hot encoding.


```Python
def catEncoding(df,ohe=True,dropFirst=False)
```

Arguments: <br><br>
*df*- dataframe to be used (pandas.DataFrame).

*ohe*- Method used for converting categorical columns (Boolean)
                 
       Possible values- True for one hot encoding, False for label encoding


*dropFirst*- Method to determine whether dummy variable is to be discarded or not. Valid only if ohe=True (Boolean).
                 
       Possible values- True for dropping dummy variable, otherwise False

Returns: <br><br>*Dataframe with all numerical columns.*


#### 3) **remOutliers()** - Remove outliers from the dataset using number of standard deviations (z-score).


```Python
def remOutliers(df,n_std=3)
```

Arguments: <br><br>
*df*- dataframe to be used (pandas.DataFrame).

*n_std*- The number of standard deviations upto which the values are to be kept (float).
                 
       Default - 3.0

Returns: <br><br>*Dataframe without outliers.*

#### 4) **scaleVals()** - Scale the values present in the dataset


```Python
def scaleVals(df,scale_list=None,scale_type='std')
```

Arguments: <br><br>
*df*- dataframe to be used (pandas.DataFrame).

*scale_list*- list of all columns to be scaled. Default: all columns (list).
 
*scale_type*- Method used for scaling (string)
                 
       Possible values- 'std' for standard scaling (default) and 'minmax' for min-max scaling

Returns: <br><br>*Dataframe with scaled values.*

#### 5) **testSplit()** - Split the dataset into training and test set


```Python
def testSplit(X,y,test_split=0.2)
```

Arguments: <br><br>
*X*- dataframe with all the features (pandas.DataFrame/ numpy.array).

*y*- target column (pandas.DataFrame/ numpy.array). 
 
*test_split*- Ratio of test set to the the total data (float). Default: 0.2
                 
Returns: <br><br>4 Dataframes - <br> *X_train,X_test,y_train,y_test*


#### 6) **splitAndTrain()** - Split the dataset into train and test test, train a random forest model on the training set and save the model as a pickle file.


```Python
def splitAndTrain(X,y,test_split=0.2,folds=5,task_type='c',model_name='model')
```

Arguments: <br><br>
*X* - dataframe with all the features (pandas.DataFrame/ numpy.array).

*y* - target column (pandas.DataFrame/ numpy.array).

*scale_list* - list of all columns to be scaled (list). Default: all columns.
 
*test_split* - Ratio of test set to the the total data (float). Default: 0.2

*folds* - number of folds for k-fold cross validation (int). Default: 5

*task_type* - type of task to be carried out by the random forest model (string).
                 
       Possible values- 'c' for classification (default) and 'r' for regression

*model_name* - name of the model pkl file to be saved (string). Default: 'model'

Returns: <br><br>
*score* - accuracy of the model trained. Accuracy for classification and R2 score for regression.
*model* - sklearn model (Random Forest) trained by the function on the given dataset.


#### 7) **predScore()** - Score the models predictions


```Python
def predScore(y_true,y_pred,task_type='c'):
```

Arguments: <br><br>
*y_true*- Vector of actual labels from the test set (numpy.array)

*y_pred*- Vector of predictions made by the model (numpy.array).
 
*task_type* - type of task that was carried out (string).
                 
       Possible values- 'c' for classification (default) and 'r' for regression
                 
Prints: <br><br> *Prints the accuracy value for classification and MSE score for regression.*

### Main Functions:

#### 1) **trainPipeline()** - Main function to run the complete data preprocessing and model training pipeline. Automatically replace missing values, encode categorical features, remove outliers, scale values, split the dataset and train the model.


```Python
def trainPipeline(dataframe,features,target,na_method='drop',ohe=True,
                      dropFirst=False,rem_outliers=True,n_std=3,scale_vals=True,
                      scale_list=None,scale_type='std',test_split=0.2,folds=5,model_name='model')

```

Arguments: <br><br>
*dataframe* - the pandas dataframe to be used for processing (pandas.DataFrame)

*features* - list of columns in the dataframe to be used as features (list)

*target* - name of target column (string)

*na_method*- method used to deal with the missing values (string).
                 
       Possible values- 'drop' (default), 'mode' and 'mean'

*ohe* - Method used for converting categorical columns (Boolean)
                 
       Possible values- True for one hot encoding, False for label encoding


*dropFirst* - Method to determine whether dummy variable is to be discarded or not. Valid only if ohe=True (Boolean).
                 
       Possible values- True for dropping dummy variable, otherwise False

*rem_outliers* -  whether outliers are to be removed (Boolean)
                 
       Possible values- True for removing outliers (Default), otherwise False

*n_std*- The number of standard deviations upto which the values are to be kept (int)
                 
       Default - 3

*scale_vals* -  whether values are to be scaled or not (Boolean)
                 
       Possible values- True for scaling (Default), False for no scaling

*scale_list*- list of all columns to be scaled. Default: all columns (list).
 
*scale_type*- Method used for scaling (string)
                 
       Possible values- 'std' for standard scaling (default) and 'minmax' for min-max scaling

*test_split* - Ratio of test set to the the total data (float). Default: 0.2

*folds* - number of folds for k-fold cross validation (int). Default: 5

*task_type* - type of task to be carried out by the random forest model (string).
                 
       Possible values- 'c' for classification (default) and 'r' for regression

*model_name* - name of the model pkl file to be saved (string). Default: 'model'

Returns: <br><br>
*model* - sklearn model (Random Forest) trained on the given dataset.

*X* - Preprocessed dataframe used to train model

*y* - target vector


#### 2) **predictPipeline()** - Main function to run the complete prediction pipeline using the model trained with the trainPipeline() function. The trainPipeline() function must be executed before using the predictPipeline().


```Python
def predictPipeline(dataframe,features,na_method='drop',ohe=True,
                      dropFirst=False,rem_outliers=True,n_std=3,scale_vals=True,
                      scale_list=None,scale_type='std',model_name='model'):

```
**Arguments for predictPipeline() must be identical to trainPipeline() to ensure that the processed dataframes are identical in both cases.** <br><br> 
Arguments: <br><br>
*dataframe* - the pandas dataframe to be used for predictions (pandas.DataFrame)

*features* - list of columns in the dataframe to be used as features (list)

*model_name* - name of the model saved in the trainPipeline() (string).

** Remaining arguments are identical to trainPipeline().**

Returns: <br><br>
*pred* - array of predictions made using the given dataframe and model.

#### 3) **processDf()** - Function to preprocess the dataframe. Similar to trainPipeline() except no model is trained and returned. 


```Python
def processDf(dataframe,features,target,na_method='drop',ohe=True,
                  dropFirst=False,rem_outliers=True,n_std=3,scale_vals=True,
                  scale_list=None,scale_type='std')

```

Arguments: <br><br>
Arguments are identical to trainPipeline().

Returns: <br><br>
*X* - Preprocessed dataframe for model training

*y* - target vector

#### 4) **processAndSplit()** - Function to preprocess the dataframe and split into train and test set. Similar to processDF() except the processed dataframe is split and returned. 


```Python
def processAndSplit(dataframe,features,target,na_method='drop',ohe=True,
                  dropFirst=False,rem_outliers=True,n_std=3,scale_vals=True,
                  scale_list=None,scale_type='std',test_split=0.2):

```

Arguments: <br><br>
Arguments are identical to trainPipeline().

Returns: <br><br>4 Dataframes - <br> *X_train,X_test,y_train,y_test*


##### Copyright (c) 2021 Karan-Malik














