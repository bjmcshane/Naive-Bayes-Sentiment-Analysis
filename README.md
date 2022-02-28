# Naive-Bayes-Sentiment-Analysis
Naive bayes sentiment analysis perormed using both maximum likelihood and maximum a posteriori approaches.

This was a programming project in my graduate level machine learning class at Indiana University. It's very similar to spam vs non-spam email classification Naive Bayes programs I've written in the past, here however I was working with positive and negative move reviews from amazon, imdb, and yelp. This assignment also required the use of cross-validation.

Check out this [link](https://towardsdatascience.com/a-hitchhikers-guide-to-sentiment-analysis-using-naive-bayes-classifier-b921c0fb694) for a more in-depth explanation of using Naive Bayes for sentiment analysis!

## Results
### Average accuracy as a function of training set size
For this part, I had to run stratefied cross validation to generate learning curves for my Naive Bayes classifier on a variety of different training set sizes (the 0.2 mark displays the accuracy when using 20% of the original training set). m=0 corresponds to the maximum likelihood approach and m=1 corresponds to the MAP estimate.


![alt text](https://github.com/bjmcshane/Naive-Bayes-Sentiment-Analysis/blob/main/results/images/part1_imdb.png?raw=true)
![alt text](https://github.com/bjmcshane/Naive-Bayes-Sentiment-Analysis/blob/main/results/images/part1_yelp.png?raw=true)
![alt text](https://github.com/bjmcshane/Naive-Bayes-Sentiment-Analysis/blob/main/results/images/part1_amzn.png?raw=true)

### Average accuracy as a function of the m parameter
---
Very similar idea here, except we're varying the m parameter instead of the training set size. Here, m=0 still corresponds to the maximum likelihood estimate and anything else still corresponds to the MAP estimate, but the higher the value of m the more "smoothing" that's used to avoid 0/1 extreme solutions.


![alt text](https://github.com/bjmcshane/Naive-Bayes-Sentiment-Analysis/blob/main/results/images/part2_imdb.png?raw=true)
![alt text](https://github.com/bjmcshane/Naive-Bayes-Sentiment-Analysis/blob/main/results/images/part2_yelp.png?raw=true)
![alt text](https://github.com/bjmcshane/Naive-Bayes-Sentiment-Analysis/blob/main/results/images/part2_amzn.png?raw=true)
