#This program reads the tweets into data frame and computes sentiment after preprocessing.
#It then determines bigram words which are occuring together. Then pie chart for bigram words is displayed.
#A network of connect words is created and visualized.
import re
import pandas as pd 
import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt 
import warnings 
import os
from textblob import TextBlob
from nltk.stem import PorterStemmer
from nltk import bigrams
import collections


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


#computer the sentiment of tweets using Textblob
data['cleanT']=data['tweet'].apply(preprocess_tweet)
data['sentiment']=data['cleanT'].map(get_sentiment)
data['tokens']=data['cleanT'].apply(lambda x:x.split())
data['stemmed']=data['tokens'].apply(lambda x: [porter.stem(i) for i in x])
data['cleanStemmedT']=data['stemmed'].apply(lambda x:' '.join(i for i in x))
data['polarity']=data['sentiment'].apply(lambda x: x.polarity)

#bigram words which are words occuring in together.
bigramWords=[list(bigrams(tweets.split())) for tweets in data['cleanStemmedT']]
print(bigramWords[0])


bigramsFlatten=[pairWords for pairWords in bigramWords ]
bigramsFlatten=sum(bigramsFlatten,[])
bigramFreqs=collections.Counter(bigramsFlatten)
bigramMostCommon20=bigramFreqs.most_common(20)
df_bigramMostCommon20=pd.DataFrame(bigramFreqs.most_common(20),columns=['bigramWords','frequency'])
df_bigramMostCommon20Dict=df_bigramMostCommon20.set_index('bigramWords').T.to_dict('records')

#Pie chart
fig,ax=plt.subplots()
ax.axis('equal')

pieBigramWords,_,_=ax.pie(list(df_bigramMostCommon20['frequency']),radius=1.3, labels=list(df_bigramMostCommon20['bigramWords']),autopct='%1.2f%%')
plt.setp( pieBigramWords, width=0.3, edgecolor='white')
plt.show()


#graph of bigram words from tweets
graphOfWords=nx.Graph()

for k,v in df_bigramMostCommon20Dict[0].items():
    print(k[0],'-->',k[1])
    graphOfWords.add_edge(k[0],k[1],weight=(v*10))

graphOfWords.add_node("Airline",weight=100)

fig, ax = plt.subplots(figsize=(10, 10))
position=nx.spring_layout(graphOfWords, k=1)
nx.draw_networkx(graphOfWords,position,ax=ax,with_labels = False)
for key, value in position.items():
    x, y = value[0]+0.025, value[1]+0.05
    ax.text(x, y, s=key,bbox=dict(facecolor='yellow', alpha=0.25), horizontalalignment='center', fontsize=12)
plt.show()