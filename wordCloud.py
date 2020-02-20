#Word cloud for airline tweets
import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import nltk
import warnings 
import os
from textblob import TextBlob
from nltk.stem import PorterStemmer
from wordcloud import WordCloud

warnings.filterwarnings("ignore", category=DeprecationWarning)

#os.chdir('Specify the path')
brand='easyjet'
porter=PorterStemmer()

#preprocess the tweets
def preprocess_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

#obtain the sentiment of tweets
def get_sentiment(tweet):
    ana=TextBlob(tweet)
    return(ana.sentiment)

data=pd.read_csv('easyjet.txt', sep=';' , header=None)
data.drop(data.index[0],axis=0,inplace=True)
data.drop(data.columns[2],axis=1,inplace=True)

data.columns=['date','tweet']
data['sentiment']=np.nan

data['cleanT']=data['tweet'].apply(preprocess_tweet)
data['sentiment']=data['cleanT'].map(get_sentiment)
data['tokens']=data['cleanT'].apply(lambda x:x.split())
data['stemmed']=data['tokens'].apply(lambda x: [porter.stem(i) for i in x])
data['cleanStemmedT']=data['stemmed'].apply(lambda x:' '.join(i for i in x))
data['polarity']=data['sentiment'].apply(lambda x: x.polarity)

#word cloud for all words
allAirlineWords=' '.join([tweet for tweet in data['cleanStemmedT']])
wordCloudAirline=WordCloud(width=800, height=500, random_state=100, max_font_size=110).generate(allAirlineWords)
plt.imshow(wordCloudAirline)
plt.show()

#word cloud for positive tweet words
positiveAirlineWords=' '.join([tweet for tweet in data['cleanStemmedT'][data['polarity']>0]])
wordCloudAirlinePositive=WordCloud(width=800, height=500, random_state=100, max_font_size=110).generate(positiveAirlineWords)
plt.imshow(wordCloudAirlinePositive)
plt.show()