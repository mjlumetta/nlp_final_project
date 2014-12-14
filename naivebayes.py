"""
Code for naive Bayes classifier.

Mike Lumetta, Chris Miller
"""
from declist import get_feature_counts, get_features 
from operator import itemgetter
from math import log

def build_classifier(trainData, stopwords=False, caseFolding=False):
    sentiments = get_sentiment_probabilities(trainData)
    feature_counts = get_feature_counts(trainData, 'tweets', stopwords, caseFolding)
    feature_counts = smooth_counts(feature_counts, 0.1)
    feature_probs = get_feature_probs(feature_counts)
    return (sentiments, feature_probs)

def get_sentiment_probabilities(data):
    counts = {}
    instances = 0
    for keyword in data.keys():
        for instance in data[keyword].keys():
            instances += 1
            for sen in data[keyword][instance]['answers']:
                if sen not in counts:
                    counts[sen] = 1
                else:
                    counts[sen] += 1
    for sen in counts:
        counts[sen] = float(counts[sen]) / float(instances)
    return counts

def smooth_counts(counts, alpha):
    features = get_all_features(counts)
    for sen in counts:
        for f in features:
            if f not in counts[sen]:
                counts[sen][f] = 0.1
            else:
                counts[sen][f] += 0.1
        # Add a minimum count for each sentiment for unseen features in test data
        counts[sen]['UNKNOWN'] = 0.1
    return counts

def get_all_features(feature_counts):
    features = []
    for sen in feature_counts:
        for f in feature_counts[sen]:
            features.append(f)
    return set(features)

def get_feature_probs(feature_counts):
    probs = {}
    for sen in feature_counts:
        probs[sen] = {}
        total = 0
        for key in feature_counts[sen]:
            total += feature_counts[sen][key]
        for key in feature_counts[sen]:
            probs[sen][key] = feature_counts[sen][key] / float(total)
    return probs

def classifyNBC(instance, classifier, caseFolding=True, stopwordList=[]):
    scores = {}
    sentiment_probs = classifier[0]
    feature_probs = classifier[1]
    features = get_features(instance, range(2,4), caseFolding, stopwordList, 3)
    for sen in sentiment_probs:
        logScore = log(sentiment_probs[sen])
        for f in features:
            if f in feature_probs[sen]:
                logScore += log(feature_probs[sen][f])
            else:
                logScore += log(feature_probs[sen]['UNKNOWN'])
        scores[sen] = logScore
    ordered_scores = sorted(scores.items(), key=itemgetter(1), reverse=True)
    return ordered_scores[0][0]
