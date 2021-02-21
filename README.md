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
*df*- dataframe to be used.

*na_method*- method used to deal with the missing values.
                 
       Possible values- 'drop' (default), 'mode' and 'mean'

Returns: <br><br>*Dataframe without missing values.*

<br>


#### 2) **catEncoding()** - Converting categorical columns to numerical, using label encoding or one-hot encoding.


```Python
def catEncoding(df,ohe=True,dropFirst=False)
```

Arguments: <br><br>
*df*- dataframe to be used.

*ohe*- Method used for converting categorical columns
                 
       Possible values- True for one hot encoding, False for label encoding


*dropFirst*- Method to determine whether dummy variable is to be discarded or not. Valid only if ohe=True.
                 
       Possible values- True for dropping dummy variable, otherwise False

Returns: <br><br>*Dataframe with all numerical columns.*


#### 3) **remOutliers()** - Remove outliers from the dataset using number of standard deviations (z-score).


```Python
def remOutliers(df,n_std=3)
```

Arguments: <br><br>
*df*- dataframe to be used.

*n_std*- The number of standard deviations upto which the values are to be kept
                 
       Default - 3

Returns: <br><br>*Dataframe without outliers.*

#### 4) **scaleVals()** - Scale the values present in the dataset


```Python
def scaleVals(df,scale_list=None,scale_type='std')
```

Arguments: <br><br>
*df*- dataframe to be used.

*scale_list*- list of all columns to be scaled. Default: all columns.
 
*scale_type*- Method used for scaling
                 
       Possible values- 'std' for standard scaling (default) and 'minmax' for min-max scaling

Returns: <br><br>*Dataframe with scaled values.*

#### 5) **testSplit()** - Split the dataset into training and test set


```Python
def testSplit(X,y,test_split=0.2)
```

Arguments: <br><br>
*X*- dataframe with all the features.

*y*- target column.
 
*test_split*- Ratio of test set to the the total data. Default: 0.2
                 
Returns: <br><br>4 Dataframes - <br> *X_train,X_test,y_train,y_test*


#### 6) **splitAndTrain()** - Split the dataset into train and test test, train a random forest model on the training set and save the model as a pickle file.


```Python
def splitAndTrain(X,y,test_split=0.2,folds=5,task_type='c',model_name='model')
```

Arguments: <br><br>
*X* - dataframe with all the features.

*y* - target column.

*scale_list* - list of all columns to be scaled. Default: all columns.
 
*test_split* - Ratio of test set to the the total data. Default: 0.2

*folds* - number of folds for k-fold cross validation. Default: 5

*task_type* - type of task to be carried out by the random forest model.
                 
       Possible values- 'c' for classification (default) and 'r' for regression

*model_name* - name of the model pkl file to be saved. Default: 'model'

Returns: <br><br>
*score* - accuracy of the model trained. Accuracy for classification and R2 score for regression.
*model* - sklearn model (Random Forest) trained by the function on the given dataset.


#### 7) **predScore()** - Score the models predictions


```Python
def predScore(y_true,y_pred,task_type='c'):
```

Arguments: <br><br>
*y_true*- Vector of actual labels from the test set (y_test)

*y_pred*- Vector of predictions made by the model.
 
*task_type* - type of task that was carried out.
                 
       Possible values- 'c' for classification (default) and 'r' for regression
                 
Prints: <br><br> *Prints the accuracy value for classification and MSE score for regression.*

### Main Functions:










