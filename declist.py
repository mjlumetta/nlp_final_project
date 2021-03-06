"""
Code for building an updated decision list to handle tweets as opposed to
single lexical elements. Uses some of the same code as the previous week's
lab. If we were building this code to be used again and again (i.e., in
production), a better strategy would be to make an abstract class that could
with child classes that implement each.

NEW THIS WEEK: modified to include ngrams and skipgrams.

Mike Lumetta, Chris Miller
"""

from parseTweet import parse_tweets
from math import log
from operator import itemgetter
from helpers import negation_processing

def build_decision_list_file(filename):
    tweetData = parse_tweets(filename, 'B')
    return build_decision_list(tweetData, 'tweets')

def build_decision_list(tweetData, keyword, stopwords=False, caseFolding=False):
    counts = get_feature_counts(tweetData, keyword, stopwords, caseFolding)
    counts = smooth_counts(counts, 0.1)
    score_list = get_feature_scores(counts)
    decision_list = sorted(score_list, key=itemgetter(2), reverse=True)
    return decision_list

def get_feature_counts(tweetData, keyword, stopwords=False, caseFolding=False):
    data = tweetData[keyword]
    counts = {}
    if stopwords:
        stopwordList = get_stopwords(data, 0.1)
    else:
        stopwordList = []
    for instance in data.keys():
        answers = data[instance]['answers']
        features = get_features(data[instance], range(2,4), caseFolding, stopwordList, 3)
        for sentiment in answers:
            if sentiment not in counts:
                counts[sentiment] = {}
            for feature in features:
                if feature not in counts[sentiment]:
                    counts[sentiment][feature] = 1
                else:
                    counts[sentiment][feature] += 1
    return counts

def get_features(instanceData, ngramRange=range(2,4), caseFolding=True, stopwordList=[], nSkipgram=3):
    features = get_bag_of_words(instanceData, stopwordList, caseFolding)
    ngrams = {}
    for k in ngramRange:
        ngrams[k] = get_ngrams(instanceData, k, caseFolding)
        features += ngrams[k]
    features += get_skipgrams(ngrams[nSkipgram], nSkipgram, caseFolding)
    return features

def get_bag_of_words(instanceData, stopwordList=[], caseFolding=False):
    words = instanceData['words']
    if len(stopwordList) > 0:
        words = [w for w in words if w not in stopwordList]
    if caseFolding:
        words = [w.lower() for w in words]
    return words

def get_ngrams(instanceData, n, caseFolding=False):
    words = instanceData['words']
    if caseFolding:
        words = [w.lower() for w in words]
    ngrams = [words[i:i+n] for i in range(len(words) - n)]
    ngrams = [" ".join(w) for w in ngrams]
    return ngrams

def get_skipgrams(ngrams, n, caseFolding=False):
    ngrams = [gram.split() for gram in ngrams]
    for i in range(len(ngrams)):
        for j in range(1,n-1):
            ngrams[i][j] = '*'
    ngrams = [" ".join(gram) for gram in ngrams]
    return ngrams

def get_stopwords(keywordData, threshold):
    word_appearances = {}
    N = len(keywordData)
    for instance in keywordData.keys():
        word_set = set(keywordData[instance]['words'])
        for w in word_set:
            if w not in word_appearances:
                word_appearances[w] = 1
            else:
                word_appearances[w] += 1
    stopwords = []
    for w in word_appearances.keys():
        if word_appearances[w] / N >= threshold:
            stopwords.append(w)
    return stopwords

def smooth_counts(counts, alpha):
    for sense in counts:
        for feature in counts[sense]:
            counts[sense][feature] += alpha
    return counts

def get_feature_scores(counts):
    scores = []
    for sense in counts:
        for feature in counts[sense]:
            presence_elsewhere = 0
            for sense2 in counts:
                if sense2 != sense and feature in counts[sense2]:
                    presence_elsewhere += counts[sense2][feature]
            if presence_elsewhere == 0:
                presence_elsewhere = 0.1
            score = log(counts[sense][feature]) / log(presence_elsewhere)
            scores.append((sense, feature, score))
    return scores

def classify(instance, decision_list, MFS):
    # features = get_features(instance, range(2,4), True, [], 3)
    features = get_bag_of_words(instance, caseFolding=True)  
    for entry in decision_list:
        if entry[1] in features:
            return entry[0]
    return MFS

if __name__ == "__main__":
    filename = 'twitter-train-full-B.tsv'
    tweetData = parse_tweets(filename, 'B')
    stopwords = get_stopwords(tweetData['tweets'], 0.2)
    stopwords2 = get_stopwords(tweetData['tweets'], 0.1)
    print("Stopwords with threshold of 0.2\n", stopwords)
    print("Stopwords with threshold of 0.1\n", stopwords2)

    print()
    example = ["This", "is", "not", "a", "very", "good", "example", ",", "but", "that", "one", "was", "."]
    print(example)
    negation_processing(example)
    print(example)
