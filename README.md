Project 2
================
Avy Harvey
6/27/2020

  - [Introduction](#introduction)
  - [Data](#data)
  - [Reports](#reports)
  - [Automation](#automation)
  - [Conclusion](#conclusion)

## Introduction

The purpose of this project is to build models to predict the number of
social media shares that an online news article will receive.
Specifically, I will train several logistic regression models and random
forest models, use repeated 10-fold cross-validation to select a single
candidate model for each type of algorithm, then compare those candidate
models on the holdout test data set. This process will be repeated for
every day of the week on which an article can be published (i.e.,
Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday).

The data used in this project is the [Online News
Popularity](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)
data set available from the UCI Machine Learning Repository. All of the
news articles in referenced in this data set were published by
[Mashable](https://www.mashable.com), and can be referenced using the
`url` variable. The data dictionary is available on the UCI website for
the data set linked above.

## Data

The data set contains 58 predictor variables and one target variable
(`shares`). The predictor variables are either numeric or binary and
represent various statistics associated with the news articles.

As mentioned before, I will be creating a different model for each day
of the week, as denoted by the `weekday_is_*` indicator variables.
Therefore, these variables (along with `is_weekend`) will not be
included in my model-building process.

The other predictor variables attempt to measure a number of things.
Some metadata about the article is available, such as number of links,
images, and videos. Additionally, the channel (or segment) that the
article falls under is also available as a set of indicator variables.
These include channels such as lifestyle, entertainment, and technology.

Most of the remaining predictor variables are derived using natural
language processing techniques. These include variables that tokenize
the text and provide counts, such as number of tokens in the title,
article content, stop words, and unique tokens. There is another set of
indicator variables that were created from a generative probabilistic
model called Latent Dirichlet allocation (LDA)
([source](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)).
There are also variables related to article sentiment and polarity.
These variables describe the subjectivity of an article’s text, as well
as describing how positive or negative the language is.

## Reports

As previously mentioned, I created reports for every weekday. Each
report contains the above Introduction and Data sections, along with
summarizations and models specific to the analyzed day by filtering
the data for that particular weekday using the `weekday_is_*`
indicator variables. You can view them by using the following links
for the respective weekday:

* [Monday](weekday_is_monday.md)
* [Tuesday](weekday_is_tuesday.md)
* [Wednesday](weekday_is_wednesday.md)
* [Thursday](weekday_is_thursday.md)
* [Friday](weekday_is_friday.md)
* [Saturday](weekday_is_saturday.md)
* [Sunday](weekday_is_sunday.md)

## Automation

This code was used to create reports for all of the `weekday_is_*`
variables. It can be executed in R to recreate the reports.

``` r
# Code referenced from class notes
weekday_vars = list("weekday_is_monday", "weekday_is_tuesday", "weekday_is_wednesday",
                  "weekday_is_thursday", "weekday_is_friday", "weekday_is_saturday",
                  "weekday_is_sunday")

# Create output file names and parameter lists
output_file <- paste0(weekday_vars, ".md")
params = lapply(weekday_vars, FUN = function(x){list(day = x)})

# Create tibble of reports to create
reports <- tibble(output_file, params)

# Create reports
pwalk(reports, rmarkdown::render,
      input = "arharvey_project2.Rmd",
      output_format = rmarkdown::github_document(toc = TRUE, pandoc_args = "--webtex"))
```

## Conclusion

The results that I saw for prediction accuracy were similar to what was reported in the original academic paper referenced on the UCI website, though they did not separate their data based on weekday. This was the primary driver behind choosing to make this a classification problem rather than a regression problem; I wanted to have a reference point for the kind of prediction accuracy I should expect.

Overall, the champion Random Forest and Logistic Regression models performed similarly to each other in terms of accuracy on the holdout test data on each day. The Random Forest model beat out the Logistic Regression model on the holdout test data on every day except Saturday. During the weekdays, the accuracy was around 62-66%, but jumped to 70-75% on the weekend, with Saturday's models yielding the highest accuracy.

The Random Forest model from Thursday had the highest accuracy rate of any *weekday* model with a rate of 66.63% on the holdout test data. The Logistic Regression model from Satuday had the highest accuracy rate of any model across all days with a rate of 75.64% on the holdout test data.

Speculating behind that jump in accuracy, it could be because people may have a habit of looking at the news before or after going to work on weekdays, and interact more frequently with online news articles on those days. This could increase the variance in some of the features on those days. People don't typically work on weekends, and may be more likely to interact with other news items.

Another possible explanation could be that there are fewer news items published on weekends since that's when most industries take a break, and people may be more likely to interact with certain categories on those days.

One thing that I should note about the models that I built is that they're not easily interpretable by the average person, particularly Random Forests. While logistic regression can be interpreted with the beta values as log-odds, that interpretation can be difficult for the average person. A decision tree, on the other hand, is easier to interpret as a series of yes/no questions, but may perform worse than the chosen models in terms of predictive accuracy. As with most things, there's a trade-off.
