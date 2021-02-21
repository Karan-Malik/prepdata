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
                 
       Possible values- 'drop', 'mode' and 'mean'
       Default value- 'drop'

Returns: <br><br>Dataframe without missing values.








