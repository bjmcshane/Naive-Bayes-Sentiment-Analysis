# Naive-Bayes-Sentiment-Analysis
Naive bayes sentiment analysis perormed using both maximum likelihood and maximum a posteriori approaches.

This was a programming project in my graduate level machine learning class at Indiana University. It's very similar to spam vs non-spam email classification Naive Bayes programs I've written in the past, here however I was working with positive and negative move reviews from amazon, imdb, and yelp. This assignment also required the use of cross-validation.

## Results
### Average accuracy as a function of training set size
For this part, I had to run stratefied cross validation to generate learning curves for my Naive Bayes classifier on a variety of different training set sizes. m=0 corresponds to the maximum likelihood approach and m=1 corresponds to the MAP estimate.
![alt text](https://github.com/bjmcshane/Naive-Bayes-Sentiment-Analysis/blob/main/results/images/part1_imdb.png?raw=true)
![alt text](https://github.com/bjmcshane/Naive-Bayes-Sentiment-Analysis/blob/main/results/images/part1_yelp.png?raw=true)
![alt text](https://github.com/bjmcshane/Naive-Bayes-Sentiment-Analysis/blob/main/results/images/part1_amzn.png?raw=true)
