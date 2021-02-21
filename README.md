# Automate Data Preprocessing for Data Science Projects
Glide through the most repetitive part of Data Science, preprocessing the dataframes with [prepdata](https://github.com/Karan-Malik/prepdata). prepdata lets you train your Machine Learning models without worrying about the imperfections of the underlying dataset. Replace missing values, remove outliers, encode categorical variables, split the dataset and train your model automatically with the comprehensive functions of this library. 

### https://pypi.org/project/prepdata/0.1.0

## Usage
The library works on [Pandas](https://pandas.pydata.org/) dataframes. All the available functions have been documented below.

### Sub-functions:

1) **missingVals()** - Remove or replace missing values from the dataset


```
missingVals(df,na_method ='drop')
```

Arguments: <br><br>
*df*- dataframe to be used.

*na_method*- method used to deal with the missing values.
                 
       Possible values- 'drop' (default), 'mode' and 'mean'

Returns: <br><br>Dataframe without missing values.

<br>


2) **catEncoding()** - Converting categorical columns to numerical, using label encoding or one-hot encoding.


```
catEncoding(df,ohe=True,dropFirst=False)
```

Arguments: <br><br>
*df*- dataframe to be used.

*ohe*- Method used for converting categorical columns
                 
       Possible values- True for one hot encoding, False for label encoding


*dropFirst*- Method to determine whether dummy variable is to be discarded or not. Valid only if ohe=True.
                 
       Possible values- True for dropping dummy variable, otherwise False

Returns: <br><br>Dataframe with all numerical columns.


3) **remOutliers()** - Remove outliers from the dataset using number of standard deviations (z-score).


```
remOutliers(df,n_std=3)
```

Arguments: <br><br>
*df*- dataframe to be used.

*n_std*- The number of standard deviations upto which the values are to be kept
                 
       Default - 3

Returns: <br><br>Dataframe without outliers.








