"""
warmup.py: 
Contains the answers to questions to answers 1-5 as well as a function that 
gets the MFS for a dataset that is also used later on.
Authors: Chris Miller and Mike Lumetta
"""
from parseTweet import parse_tweets

def main():
    #Running this file will print the answers to each question to the terminal.
    filename = '/data/cs65/semeval-2015/B/train/twitter-train-full-B.tsv'
    tweetData = parse_tweets(filename, 'B')
    #Q1: How many training examples are there?
    print("(1) There are %d training examples" % len(tweetData['tweets'].keys()))

    #Q2: How often do each of the sentiment labels show up? What about if you
    #    conflate neutral/objective/objective-OR-neutral into neutral?
    print("(2) With conflation of neutral/objective turned off or on:")
    print("OFF:")
    maxKeyCOff = MFS_counter(tweetData)
    print("---------------")
    print("ON:")
    maxKeyCOn = MFS_counter(tweetData, True)
    print("---------------")

    #Q3: What is the random baseline?
    print("(3)\n\tWith conflation off: 5 sentiments, so 20% tagged correctly.\n\tWith conflation on: 3 sentiments, so 33.3% tagged correctly." )

    #Q4: What is the most frequent sentiment baseline?
    print("(4)\n\tWith conflation off: MFS = positive with 37.3% tagged correctly\n\tWith conflation on: MFS = neutral with 48.2% tagged correctly")

"""
By default, this function gets the MFS for the input data. It will conflate
neutral/objective into one category if the conflate_flag is true. It will 
return the probability for an input sentiment if the sentiment is set
"""
def MFS_counter(data, conflate_flag=False, sentiment=None):
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
    
    #if we're looking for a particular sentiment probability, return it
    if(sentiment != None):
        sentiment_prob = float(sentiment_dict[sentiment]) / (float(total_sentiments))
        return sentiment_prob

    #else, return the most frequent sentiment
    for key in sentiment_dict:
        if (float(sentiment_dict[key])/float(total_sentiments) > max_val):
            max_val = float(sentiment_dict[key]/float(total_sentiments))
            max_string = key
        return max_string

if __name__=='__main__':
    main()
