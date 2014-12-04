"""
Module containing various helper functions for classifying tweets that have
no specific affiliation (ie cross-validation, naive bayes, etc.).

Mike Lumetta, Chris Miller
"""

from copy import deepcopy

"""
This function takes an input dict and applies negation processing to it
"""
def get_negated_dict(tweetData):
    negated_tweets = {}
    #Just making sure we don't change the original tweetData
    negated_tweets = deepcopy(tweetData)
    key_list = negated_tweets['tweets'].keys()
    for key in key_list:
        #For each tweet, process it and put it back in the dict.
        negated_tweet = negation_processing(negated_tweets['tweets'][key]['words'])
        negated_tweets['tweets'][key]['words'] = negated_tweet
    return negated_tweets

def MFS_counter(data, conflate_flag=False):
    sentiment_dict = {}
    data = data['tweets']
    sentiment_dict['neutral'] = 0
    total_sentiments = 0
    conflate_list = ['neutral', 'objective', 'objective-OR-neutral']
    for key in data.keys():
        #sum all the sentiments we see
        total_sentiments += 1
        #if there's more than one answer, it has to be objective or neutral
        if(len(data[key]['answers']) > 1):
            answer = 'objective-OR-neutral'
        else: 
            #else, it's whatever it is
            answer = data[key]['answers'][0]
        if answer not in sentiment_dict:
            #if it's not in the dictionary, and should be conflated
            if((conflate_flag == True) and (answer in conflate_list)):
                sentiment_dict['neutral'] += 1
            else: 
                #if it's not in the dictionary but shouldn't be conflated
                sentiment_dict[answer] = 1
        else:
            #if it's in the dictionary but it should be conflated
            if((conflate_flag == True) and (answer in conflate_list)):
                sentiment_dict['neutral'] += 1
            else: 
                #if it's in the dictionary but shouldn't be conflated
                sentiment_dict[answer] += 1
    max_val = 0;
    max_string = ""
    
    #else, return the most frequent sentiment
    for key in sentiment_dict:
        if (float(sentiment_dict[key])/float(total_sentiments) > max_val):
            max_val = float(sentiment_dict[key]/float(total_sentiments))
            max_string = key
        return max_string

def negation_processing(wordList):
    negate = False
    punctuation = [',', '.', '?', '!', ':', ';']
    for i in range(len(wordList)):
        if wordList[i].lower() == "not":
            negate = True
        elif wordList[i] in punctuation:
            negate = False
        elif negate:
            wordList[i] = "NOT_" + wordList[i]
    return wordList
