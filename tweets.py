"""
tweet.py:
Calculates the answers to questions 6-8 using crosseval on chunks, doing MFS
analysis, and utilizing the decision list from our previous lab.
Authors: Chris Miller and Mike Lumetta
"""

from copy import deepcopy
from warmup import *
from parseTweet import *
from crossval import *
from declist import *
from naivebayes import *
# from arktweet import tokenize

def main():
    filename = '/data/cs65/semeval-2015/B/train/twitter-train-full-B.tsv'
    tweetData = parse_tweets(filename, 'B')
    k = 5
    chunk_list =  k_chunks(tweetData, k)

    #Question 5
    print ("(5)")
    chunk_MFS_analysis(chunk_list)

    #Question 6
    print("\n(6)")
    chunk_MFS_training(chunk_list, k)
    #Question 7
    print("\n(7)")
    #(a)
    print("-----Default Decision List-----")
    #build decision list
    dec_list = build_decision_list(tweetData, 'tweets')
    #evaluate it vs. the ground truth
    result = evaluate_dec_list(dec_list, tweetData)
    print(result)
    #(b)
    print("-----Decision List + Stop-Word Processing-----")
    #build decision list with stop-word processing
    stop_dec_list = build_decision_list(tweetData, 'tweets', True, False)
    stop_result = evaluate_dec_list(stop_dec_list, tweetData)
    print(stop_result)
    #(c)
    print("-----Decision List + Case Folding Processing-----")
    #build decision list with case folding processing
    cf_dec_list = build_decision_list(tweetData, 'tweets', False, True)
    cf_result = evaluate_dec_list(cf_dec_list, tweetData)
    print(cf_result)

    #Question 8
    print("\n(8)")
    print("-----Decision List + Negation Processing-----")
    #edit the dictionary with negation processing
    n_tweetData = get_negated_dict(tweetData)
    #build the decision list on that new dictionary
    n_dec_list = build_decision_list(n_tweetData, 'tweets')
    n_result = evaluate_dec_list(n_dec_list, n_tweetData)
    print(n_result)
    
    # Question 9
    print("\n(9)")
    print("-----Naive Bayes-----")
    evaluate_naive_bayes(tweetData, k)

    # Question 10
    print("\n(10)")
    print("-----Naive Bayes + Stopwords-----")
    evaluate_naive_bayes(tweetData, k, True)
    print("-----Naive Bayes + Case Folding-----")
    evaluate_naive_bayes(tweetData, k, False, True)
    print("-----Naive Bayes + Negation-----")
    evaluate_naive_bayes(n_tweetData, k)

    """
    # Question 11
    print("\n(11)")
    tokenizedData = tokenize_data(tweetData)
    print("-----Naive Bayes + Tokenizing-----")
    evaluate_naive_bayes(tokenizedData, k)
    """

"""
This function scores the decision list against the ground truth
"""
def evaluate_dec_list(decision_list, tweetData):
    MFS = MFS_counter(tweetData, True)
    correct = 0
    total = 0
    for key in tweetData['tweets'].keys():
        #For each tweet, compare the sentiment tag via dec list and truth.
        total += 1
        dl_sent = classify(tweetData['tweets'][key], decision_list, MFS)
        dict_sent = tweetData['tweets'][key]['answers']
        #Make sure that both results are conflated into Pos/Neg/Neu
        if ('objective' in dict_sent): 
            dict_sent = 'neutral'
        else:
            dict_sent = dict_sent[0]
        if(dl_sent == 'objective'):
            dl_sent = 'neutral'
        if dl_sent == dict_sent:
            correct += 1
    return (correct/total)

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

"""
This function finds the MFS for each of k chunks
"""
def chunk_MFS_analysis(chunk_list):
    #For each chunk, find the most frequent sentiment
    for i in range(len(chunk_list)):
        print("Chunk =", i)
        MFS_counter(chunk_list[i], True)

"""
This function uses the MFS of k-1 chunks to label the remaining chunk. 
It repeats the experiment in k-fold cross-validation.
"""
def chunk_MFS_training(chunk_list, k):
    #Repeat experiment k times
    average = 0
    for i in range(k):
        combined_training = {}
        training_chunks = []
        test_chunk = []
        test_dict = {}
        print("Experiment", i, ":")
        #For each chunk, assign it to either training or test
        for j in range(k):
            if j != i:
                training_chunks.append(chunk_list[j])
            else: 
                test_chunk.append(chunk_list[i])
            #Combine all the training chunks into one dataset.
            for a in range(len(training_chunks)):
                if(a == 0):
                    combined_training['tweets'] = training_chunks[a]['tweets'].copy()
                else:
                    combined_training['tweets'].update(training_chunks[a]['tweets'])
        #Find the MFS of that dataset.
        training_result = MFS_counter(combined_training, True)
        #Find the probability of that sentiment in the training set.
        test_dict = test_chunk[0].copy()
        MFS_result = MFS_counter(test_dict, True, training_result)
        print("The probability of", training_result, "on the test set yields a percentage of", float(MFS_result))
        average += float(MFS_result)
    print("\nThe average score is", average/k)

def evaluate_naive_bayes(tweetData, k, stopwords=False, caseFolding=False):
    chunks = k_chunks(tweetData, k)
    correct = []
    total = []
    for i in range(k):
        test = chunks[i]
        other = [chunks[j] for j in range(k) if j != i]
        train = combine_dictionaries(other)
        classifier = build_classifier(train, stopwords, caseFolding)
        total.append(len(test['tweets']))
        right = 0
        for instance in test['tweets']:
            answer = classifyNBC(test['tweets'][instance], classifier)
            if answer in test['tweets'][instance]['answers']:
                right += 1
        correct.append(right)
    for i in range(k):
        print("Results for chunk", i)
        print_results(correct[i], total[i], correct[i] / total[i])

def print_results(correct, total, accuracy):
    print(correct, "correct out of", total, "total")
    print("Accuracy:", accuracy)

"""
def tokenize_data(tweetData):
    instanceData = tweetData['tweets']
    tweets = []
    for key in sorted(list(instanceData.keys)):
        tweets.append(instanceData[key]['words'])
    tokenized = tokenize(tweets)
    newData = {}
    i = 0
    for key in sorted(list(instanceData.keys)):
        newData[key]['words'] = tokenized[i]
        i += 1
    newNewData = {}
    newNewData['tweets'] = newData
    return newNewData
"""

if __name__ == "__main__":
    main()
