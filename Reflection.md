# Total score 101/100

- The numbers in the report don't line up with the tables displayed (-2)
- For decision tree and random forest, not much exploration of different parameters (-2)
- Best accuracy score (+5)



Team-Project-2
================

Data
----

Use the training data below to predict the county-level `winner16` of the 2016 presidential election, classified as Democratic (Clinton) or Republican (Trump):

```{r}
train <- read_csv("https://raw.githubusercontent.com/mgelman/data/master/train.csv")
```

The first 51 variables in this data frame contain county-level demographic and economic data. The table at the end of this doc shows the dictionary that describes these variables.

``` r
> key <- read.csv("https://raw.githubusercontent.com/mgelman/data/master/county_facts_dictionary.csv")
```

### To do:

Use the training data to create a classification model for `winner16`. I'll be judging your predictions based on your accuracy.

-   Rename the main variables you are using so that they are easier to interpret
-   If possible, describe which variables are the most important predictors
-   Use validation or cross-validation with the training data to avoid overfitting your model to the (entire) training data set which could produce a poor fit to the new test data.
-   If you consider methods like k-nn, decision trees or random forests, try playing around with their parameter values (like k, cp, or mtry) to find a more accurate model.
-   You don't need to include all 51 predictors in your model/method.

### Turn in:

#### 1. **Predictions**

The following is a link to the test data set without the `winner` response.

``` r
> test_No_Y <- read_csv("https://raw.githubusercontent.com/mgelman/data/master/test_No_Y.csv")
```

Once you find a suitable classification method using the training data, use the method to predict responses to the test data. Add these predicted `pred_winner` values to the 52nd column. Use this *exact* name! Save this modified test data file as a .csv with your team's last names added to the file name:

``` r
> write_csv(test_No_Y, "test_No_Y_LastNames.csv")
```

Push this file to your GitHub repo. I will compute your accuracy before class and declare a winner in class! **The winner will get a bonus 5 points added to their score** *Make sure that you don't rearrange the order of the rows because there is not a row id given in the data to match with the actual response file!*

#### 2. **Write-up**

Produce a detailed (2-3 pages) write-up of your classification method. Describe any data cleanup that was needed and describe how you arrived at your final model/method, including any error (or other evaluation stats) measured during model/method formulation. I should be able to read your writeup and come away with a sound understanding of how to reproduce your test set predictions. If you use any classification methods not covered in class, you should explain how the methods work.

Similar to the first project, I don't want to see R code in your write up. I only want to see output. However, I will base part of my evaluation on how concise and well commented your code is. I should be able to run your code and reproduce your test set predictions. Make sure to load all packages so that the code will run on my computer as well. If in doubt, test it on a lab computer that doesn't have all the packages/settings that your own machine has.

Grading: 100 points
-------------------

I will grade your written report on a 90 point scale based on your write-up of your classification method.

I will use a 10-point scale to assess your R code:

-   readability and conciseness: is the code readable and appropriately commented? is your code concise, or could you accomplish a task in fewer steps?
-   Score examples:
    -   10 = readable and sufficient comments, code written concisely
    -   8 = readable and commented, but some parts could be written more efficiently
    -   5 = mostly readable but contains multiple portions that could be written in a more clear manner, minimal comments
    -   0 = all code could be written in a more readable manner, no comments and irrelevant commands

### Data Dictionary

``` r
> kable(key)
```

| column\_name | description                                                            |
|:-------------|:-----------------------------------------------------------------------|
| PST045214    | Population, 2014 estimate                                              |
| PST040210    | Population, 2010 (April 1) estimates base                              |
| PST120214    | Population, percent change - April 1, 2010 to July 1, 2014             |
| POP010210    | Population, 2010                                                       |
| AGE135214    | Persons under 5 years, percent, 2014                                   |
| AGE295214    | Persons under 18 years, percent, 2014                                  |
| AGE775214    | Persons 65 years and over, percent, 2014                               |
| SEX255214    | Female persons, percent, 2014                                          |
| RHI125214    | White alone, percent, 2014                                             |
| RHI225214    | Black or African American alone, percent, 2014                         |
| RHI325214    | American Indian and Alaska Native alone, percent, 2014                 |
| RHI425214    | Asian alone, percent, 2014                                             |
| RHI525214    | Native Hawaiian and Other Pacific Islander alone, percent, 2014        |
| RHI625214    | Two or More Races, percent, 2014                                       |
| RHI725214    | Hispanic or Latino, percent, 2014                                      |
| RHI825214    | White alone, not Hispanic or Latino, percent, 2014                     |
| POP715213    | Living in same house 1 year & over, percent, 2009-2013                 |
| POP645213    | Foreign born persons, percent, 2009-2013                               |
| POP815213    | Language other than English spoken at home, pct age 5+, 2009-2013      |
| EDU635213    | High school graduate or higher, percent of persons age 25+, 2009-2013  |
| EDU685213    | Bachelor's degree or higher, percent of persons age 25+, 2009-2013     |
| VET605213    | Veterans, 2009-2013                                                    |
| LFE305213    | Mean travel time to work (minutes), workers age 16+, 2009-2013         |
| HSG010214    | Housing units, 2014                                                    |
| HSG445213    | Homeownership rate, 2009-2013                                          |
| HSG096213    | Housing units in multi-unit structures, percent, 2009-2013             |
| HSG495213    | Median value of owner-occupied housing units, 2009-2013                |
| HSD410213    | Households, 2009-2013                                                  |
| HSD310213    | Persons per household, 2009-2013                                       |
| INC910213    | Per capita money income in past 12 months (2013 dollars), 2009-2013    |
| INC110213    | Median household income, 2009-2013                                     |
| PVY020213    | Persons below poverty level, percent, 2009-2013                        |
| BZA010213    | Private nonfarm establishments, 2013                                   |
| BZA110213    | Private nonfarm employment, 2013                                       |
| BZA115213    | Private nonfarm employment, percent change, 2012-2013                  |
| NES010213    | Nonemployer establishments, 2013                                       |
| SBO001207    | Total number of firms, 2007                                            |
| SBO315207    | Black-owned firms, percent, 2007                                       |
| SBO115207    | American Indian- and Alaska Native-owned firms, percent, 2007          |
| SBO215207    | Asian-owned firms, percent, 2007                                       |
| SBO515207    | Native Hawaiian- and Other Pacific Islander-owned firms, percent, 2007 |
| SBO415207    | Hispanic-owned firms, percent, 2007                                    |
| SBO015207    | Women-owned firms, percent, 2007                                       |
| MAN450207    | Manufacturers shipments, 2007 ($1,000)                                 |
| WTN220207    | Merchant wholesaler sales, 2007 ($1,000)                               |
| RTN130207    | Retail sales, 2007 ($1,000)                                            |
| RTN131207    | Retail sales per capita, 2007                                          |
| AFN120207    | Accommodation and food services sales, 2007 ($1,000)                   |
| BPS030214    | Building permits, 2014                                                 |
| LND110210    | Land area in square miles, 2010                                        |
| POP060210    | Population per square mile, 2010                                       |
