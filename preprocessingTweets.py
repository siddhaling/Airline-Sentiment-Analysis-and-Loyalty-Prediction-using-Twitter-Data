#Sentiment analysis of Airline tweets
# Preprocessing of tweets, perform stemming on the tokens of a tweets
#Using Textblob compute the sentiment of tweets 
#Visualization such as plot sentiment, mean of sentiment per week, number of tweets per week, mean of positive and negative polarities
#Column chart of mean of positive and negative polarities, top 10 positive words plot using bar chart
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

#obtain the sentiment of a tweet
def get_sentiment(tweet):
    ana=TextBlob(tweet)
    return(ana.sentiment)

#read easyjet.txt into a data frame
data=pd.read_csv('easyjet.txt', sep=';' , header=None)
data.drop(data.index[0],axis=0,inplace=True)
data.drop(data.columns[2],axis=1,inplace=True)

data.columns=['date','tweet']
print(preprocess_tweet(data.iloc[0]['tweet']))
data['sentiment']=np.nan

data['cleanT']=data['tweet'].apply(preprocess_tweet)
print(get_sentiment(data.iloc[0]['cleanT']))
data['sentiment']=data['cleanT'].map(get_sentiment)
data['tokens']=data['cleanT'].apply(lambda x:x.split())
data['stemmed']=data['tokens'].apply(lambda x: [porter.stem(i) for i in x])
data['cleanStemmedT']=data['stemmed'].apply(lambda x:' '.join(i for i in x))
data['polarity']=data['sentiment'].apply(lambda x: x.polarity)
print(data[['tweet','polarity']].head())

#sentiment for each tweet displayed in a plot
fig=plt.figure()
ax=plt.axes()
ax.plot(data['polarity'])
plt.xlabel('Tweets')
plt.ylabel('Polarity')
plt.show()

#Mean sentiment per week in a plot
data['date']=pd.to_datetime(data['date'])
countOfTweetsPerWeek=data['polarity'].groupby(data['date'].dt.week).count()
meanOfPolarityPerWeek=data['polarity'].groupby(data['date'].dt.week).mean()

#plot mean of polarity per week
fig=plt.figure()
ax=plt.axes()
ax.plot(meanOfPolarityPerWeek)
plt.xlabel('A week')
plt.ylabel('Polarity')
plt.show()

#Tweets per week
index=np.arange(len(countOfTweetsPerWeek))
plt.bar(index,countOfTweetsPerWeek)
plt.xlabel('Tweets per week')
plt.ylabel('Count')
plt.title('Frequency of tweets per week')
plt.show()

#mean of positive and negative polarities
positivePolarity=data['polarity'][data['polarity']>0]
negativePolarity=data['polarity'][data['polarity']<0]
meanPositivePolarity=positivePolarity.mean()
meanNegativePolarity=negativePolarity.mean()

#Column chart of mean of positive and negative polarities
index=np.arange(2)
plt.bar(index,[meanPositivePolarity,meanNegativePolarity])
plt.xlabel('Mean of Polarities')
plt.ylabel('Sentiment Score')
plt.xticks(index,['Positive','Negative'])
plt.title('Mean of Polarities')
plt.show()

# column chart for words in positive tweets
dataPositivePolarity=data['cleanStemmedT'][data['polarity']>0]
positiveTweetWords=[posTweet.split() for posTweet in dataPositivePolarity]
positiveTweetWords=sum(positiveTweetWords,[])
freqPosWords=nltk.FreqDist(positiveTweetWords)
dfPositiveTweetWords=pd.DataFrame({'word':list(freqPosWords.keys()),'frequency':list(freqPosWords.values())})
top10PositiveTweetWords=dfPositiveTweetWords.nlargest(n=10,columns='frequency')

#top 10 positive words plot using bar chart
index=np.arange(len(top10PositiveTweetWords))
plt.bar(index,top10PositiveTweetWords['frequency'])
plt.xlabel('Positive tweet words')
plt.ylabel('Frequency')
plt.xticks(index,top10PositiveTweetWords['word'])
plt.title('Words in Positive tweets Analysis')
plt.show()