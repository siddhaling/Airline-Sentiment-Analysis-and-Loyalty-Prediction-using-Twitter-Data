# please mention consumer key, consumer secret, access token, access token secre
import tweepy


global CONSUMER_KEY
CONSUMER_KEY = 'CONSUMER_KEY'
global CONSUMER_SECRET
CONSUMER_SECRET = 'CONSUMER_SECRET'
global ACCESS_TOKEN
ACCESS_TOKEN = 'ACCESS_TOKEN'
global ACCESS_TOKEN_SECRET
ACCESS_TOKEN_SECRET = 'ACCESS_TOKEN_SECRET'

global auth
#obtain twitter handler
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

global api
api = tweepy.API(auth)

