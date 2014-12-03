#!/usr/bin/env python

"""
Mike Lumetta, Chris Miller
Starting point code for the decision list classifier in Lab 06
"""
from parse import getData
from math import log
from operator import itemgetter
from warmup import most_frequent_sense

def main():
    decision_lists = {}
    trainingFile = '/data/cs65/senseval3/train/EnglishLS.train'
    trainData = getData(trainingFile)
    testFile = '/data/cs65/senseval3/test/EnglishLS.test'
    testData = getData(testFile)
    k = 10
    
    for word in trainData.keys():
        decision_lists[word] = build_decision_list(trainData, word, k)

    total = 0
    correct = 0
    correct_less = 0
    correct_cutoff = 0
    for word in testData.keys():
        MFS = most_frequent_sense(trainData, word)
        modified_declist = [x for x in decision_lists[word] if x[2] > 1.0]
        for instance in testData[word].keys():
            total += 1
            instanceData = testData[word][instance]
            classification = classify(instanceData, decision_lists[word], k, MFS)
            less_rules = classify(instanceData, decision_lists[word][:100], k, MFS)
            cutoff = classify(instanceData, modified_declist, k, MFS)
            if classification in instanceData['answers']:
                correct += 1
            if less_rules in instanceData['answers']:
                correct_less += 1
            if cutoff in instanceData['answers']:
                correct_cutoff += 1
    accuracy = float(correct) / float(total)
    accuracy_less = float(correct_less) / float(total)
    accuracy_cutoff = float(correct_cutoff) / float(total)
    print "Accurately classified %f of all words" % (accuracy)
    print "%d correct of %d total" % (correct, total)
    print "With 100 rules, accurately classified %f of all words" % (accuracy_less)
    print "%d correct of %d total" % (correct_less, total)
    print "With cutoff at 1.0, Accurately classified %f of all words" % (accuracy_cutoff)
    print "%d correct of %d total" % (correct_cutoff, total)

    """
    12. Got 0.593 accuracy, slightly better than the 0.571 accuracy
    of the MFS baseline.
    14. The modified classifications perform approximately the same; see 
    output
    """

    # print decision_lists['organization.n'][:100]
    """
    13. Some of the problematic rules include:
        - pronouns... "he", "his"
        - prepositions... "between"
        - other generic words that don't have to do with organization
    """

def get_frequent_words(training, cutoff):
    return []

def classify(instance, decision_list, k, MFS):
    features = get_bag_of_words(instance, k) + get_collocations(instance)
    for entry in decision_list:
        if entry[1] in features:
            return entry[0]
    return MFS

def build_decision_list(training, word, k):
    counts = get_feature_counts(training, word, k)
    counts = smooth_counts(counts, 0.1)
    score_list = get_feature_scores(counts)
    decision_list = sorted(score_list, key=itemgetter(2), reverse=True)
    return decision_list

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


"""
smooth_counts
"""
def smooth_counts(counts, alpha):
    for sense in counts:
        for feature in counts[sense]:
            counts[sense][feature] += alpha
    return counts

"""
get_feature_counts
Building feature counts dictionary for a given word.

Top-level
Keys: senses of word
Values: nested dictionaries

Nested dictionaries, organized by sense
Keys: features
    bag of words are represented as just that word
    collocations are the given two word phrases
Values: feature counts
"""
def get_feature_counts(training, word, k):
    wordData = training[word]
    counts = {}
    for instance in wordData.keys():
        answers = wordData[instance]['answers']
        bag = get_bag_of_words(wordData[instance], k)
        collocations = get_collocations(wordData[instance])
        for sense in answers:
            if sense not in counts:
                counts[sense] = {}
            for feature in bag + collocations:
                if feature not in counts[sense]:
                    counts[sense][feature] = 1
                else:
                    counts[sense][feature] += 1
    return counts
            

"""
get_bag_of_words
Returns a list of words that constitute the bag of words
"""
def get_bag_of_words(instanceData, k):
    bag = []
    for index in instanceData['heads']:
        if index - k >= 0:
            bag += instanceData['words'][index-k:index]
        else:
            bag += instanceData['words'][0:index]
        word_len = len(instanceData['words'])
        if index + k < word_len:
            bag += instanceData['words'][index+1:index+k+1]
        else:
            bag += instanceData['words'][index+1:]
    return bag  

"""
get_collocations
Returns all collocations for a given instance.
"""
def get_collocations(instanceData):
    collocations = []
    words = instanceData['words']
    for index in instanceData['heads']:
        if index != 0:
            collocations.append(words[index-1] + " " + words[index])
        if index != len(words) - 1:
            collocations.append(words[index] + " " + words[index + 1])
    return collocations


if __name__=='__main__':
    main()
