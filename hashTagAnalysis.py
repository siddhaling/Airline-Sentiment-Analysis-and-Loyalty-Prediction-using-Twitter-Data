#This program perform the hash tag analysis,finds the positive hash tags and negative hash tags
#The frequency of positive hash tags and negative hash tags are determined and plotted using bar chart
import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import nltk
import warnings 

import os
from textblob import TextBlob
from nltk.stem import PorterStemmer


warnings.filterwarnings("ignore", category=DeprecationWarning)

#os.chdir('Specify current directory')
brand='easyjet'
porter=PorterStemmer()
#preprocess the tweets
def preprocess_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

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



#Hashtag analysis
def collect_hashtags(text):
    hts = re.findall(r"#(\w+)", text)
    return hts

#Positive Hash Tag Analysis
data['hashTagsPositive']=data['tweet'][data['polarity']>0].apply(collect_hashtags)

hashTagsForPositiveTweets=data['hashTagsPositive'][data['hashTagsPositive'].notnull()]
allHashTagsPositive=list(hashTagsForPositiveTweets)
allHashTagsPositive=sum(allHashTagsPositive,[])

#Positive Hash Tag Analysis
freqTags = nltk.FreqDist(allHashTagsPositive)
dfFreqTags=pd.DataFrame({'PositiveHashTags':list(freqTags.keys()), 'frequency':list(freqTags.values())})
dfFreqTagsTop10=dfFreqTags.nlargest(n=10,columns='frequency')
index=np.arange(len(dfFreqTagsTop10))
plt.bar(index,dfFreqTagsTop10['frequency'])
plt.xlabel('HasTags')
plt.ylabel('Frequency')
plt.xticks(index,dfFreqTagsTop10['PositiveHashTags'])
plt.title('Positive Hash Tag Analysis')
plt.show()

#Negative Hash Tag Analysis
data['hashTagsNegative']=data['tweet'][data['polarity']<0].apply(collect_hashtags)
hashTagsForNegativeTweets=data['hashTagsNegative'][data['hashTagsNegative'].notnull()]
allHashTagsNegative=list(hashTagsForNegativeTweets)
allHashTagsNegative=sum(allHashTagsNegative,[])
negFreqTags=nltk.FreqDist(allHashTagsNegative)
dfNegFreqTags=pd.DataFrame({'NegativeHashTags':list(negFreqTags.keys()),'frequency':list(negFreqTags.values())})
dfNegFreqTagsTop10=dfNegFreqTags.nlargest(n=10,columns='frequency')
plt.bar(index,dfNegFreqTagsTop10['frequency'])
plt.xlabel('HasTags')
plt.ylabel('Frequency')
plt.xticks(index,dfNegFreqTagsTop10['NegativeHashTags'])
plt.title('Negative Hash Tag Analysis')
plt.show()
