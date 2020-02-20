#Connect to twitter account using authentication keys.
#Then using search keyword for search for tweets. Write the tweets into text file
from Auth import *
import tweepy as tw
import os

#Consumer key, secrete, access token, token secret are collected from Twitter account
consumer_key= CONSUMER_KEY
consumer_secret= CONSUMER_SECRET
access_token= ACCESS_TOKEN
access_token_secret= ACCESS_TOKEN_SECRET

#Authenticate with keys
authemtication = tw.OAuthHandler(consumer_key, consumer_secret)
authemtication.set_access_token(access_token, access_token_secret)
api = tw.API(authemtication, wait_on_rate_limit=True)


searchKeyWord= "@easyjet"
#search for given keyword from the date 2018-11-01 and 100 tweets are collected
tweetsOfKeyword = tw.Cursor(api.search, q=searchKeyWord, lang="en",since='2018-11-01').items(100)

#os.chdir('Specify the path')

#write the twitter into a text file
fwriter=open('easyjet.txt','w')
for tweet in tweetsOfKeyword:
    print(str([tweet.created_at.strftime("%Y-%m-%d %H:%M"), tweet.text]))
    fwriter.write(tweet.created_at.strftime("%Y-%m-%d %H:%M")+' '+tweet.text+'\n')
fwriter.close()