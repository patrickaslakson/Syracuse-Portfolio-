import json
import csv
import tweepy
import re
import os

os.getcwd()
os.chdir(r'C:\School\Text Mining\Twitter')

def search_for_hashtags(consumer_key, consumer_secret, access_token, access_token_secret, hashtag_phrase):
    # create authentication for accessing Twitter
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    # initialize Tweepy API
    api = tweepy.API(auth)

    # get the name of the spreadsheet we will write to
    #fname = '_'.join(re.findall(r"#(\w+)", hashtag_phrase))
    fname='selfdrivingcartweets'

    # open the spreadsheet we will write to
    with open('%s.csv' % (fname), 'w', encoding='utf8') as file:
        w = csv.writer(file)

        # write header row to spreadsheet
        w.writerow(['timestamp', 'tweet_text', 'username', 'all_hashtags', 'followers_count'])

        # for each tweet matching our hashtags, write relevant info to the spreadsheet
        for tweet in tweepy.Cursor(api.search, q=hashtag_phrase + ' -filter:retweets', \
                                   lang="en", tweet_mode='extended').items(100):
            w.writerow([tweet.created_at, tweet.full_text.replace('\n', ' '),
                        tweet.user.screen_name,
                        [e['text'] for e in tweet._json['entities']['hashtags']], tweet.user.followers_count])


consumer_key = 'Insert key here'
consumer_secret = 'Insert key here'
access_token = 'Insert key here'
access_token_secret = 'Insert key here'

#hashtag_phrase = '#driverless OR #autonomous OR #engineering OR #driverlesscars OR #formulastudent OR #racecar OR #electric OR #technology OR #selfdriving OR #ai OR #autonomousvehicles OR #cars OR #fsae OR #selfdrivingcars OR #team OR #racing OR #motorsport OR #robot OR #fsg OR #robots OR #becauseracecar OR #ev OR #fsa OR #dv OR #iot OR #future OR #fs OR #formulastudentgermany OR #tuhamburg OR #bhfyp'
hashtag_phrase = '#autonomous OR #driverless OR #driverlesscars OR #autonomousvehicles OR #selfdriving OR #selfdrivingcars'

#tagList=['#driverless', '#autonomous', '#engineering', '#driverlesscars', '#formulastudent', '#racecar', '#electric', '#technology', '#selfdriving', '#ai', '#autonomousvehicles', '#cars', '#fsae', '#selfdrivingcars', '#team', '#racing', '#motorsport', '#robot', '#fsg', '#robots', '#becauseracecar', '#ev', '#fsa', '#dv', '#iot', '#future', '#fs', '#formulastudentgermany', '#tuhamburg', '#bhfyp']

if __name__ == '__main__':
#for hastag_phrase in tagList:
    search_for_hashtags(consumer_key, consumer_secret, access_token, access_token_secret, hashtag_phrase)