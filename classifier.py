"""
Mike Lumetta, Chris Miller

Implementation of our improved sentiment tagger based on the TeamX paper.
Adds ngrams and skipgrams to the list of features. For this one, we will
only use the decision list. Uses negation processing. 

We only  go up to trigrams rather than 4-grams for the sake of time. Also,
we allow the user to pick a segment from 1 to k to perform cross-validation
as a convenience method. We can still do true cross-validation by running it 
k times. Or, it could easily be extended but would take a while.

The implementation of a naive Bayes classifier could do act similarly, taking
the new ngrams and skipgrams as features.

The improved decision list offers minor improvement over the old one in most 
cases. However, the improvement is largely negligible. We would need more of
the extensive pre-processing that TeamX performed to really do well; ngrams
in isolation don't make much of a difference.
"""
#!/usr/bin/python3

import declist
import declist2
import sys
from parseTweet import *
from crossval import *
from tweets import get_negated_dict
from warmup import *
from scorer import *

def main():
    try:
        k = int(sys.argv[2])
        flag = True
    except ValueError:
        flag = False

    trainData = parse_tweets(sys.argv[1], 'B')
    if flag:
        chunks = k_chunks(trainData, k)
        i = int(input("Pick a test segment from 0 to k-1: "))
        if i not in range(k):
            return -1
        else:
            crossvalTest(chunks, i)
    else:
        testVsTrain(trainData, sys.argv[2])

def crossvalTest(chunks, i):
    testData = chunks[i]
    trainSet = [chunks[j] for j in range(len(chunks)) if j != i]
    trainData = combine_chunks(trainSet)
    stats = test(trainData, testData)
    oldScore = scorer(stats[0])
    newScore = scorer(stats[1])

    print("The old version got an official score of", oldScore)
    print("The new version got an official score of", newScore)



def combine_chunks(chunks):
    combined = {}
    combined['tweets'] = chunks[0]['tweets'].copy()
    for i in range(1, len(chunks)):
        combined['tweets'].update(chunks[i]['tweets'])
    return combined

def testVsTrain(trainData, filename):
    testData = parse_tweets(filename, 'B')
    stats = test(trainData, testData)
    oldScore = scorer(stats[0])
    newScore = scorer(stats[1])

    print("The old version got an official score of", oldScore)
    print("The new version got an official score of", newScore)


def test(trainData, testData):
    # old version
    old = declist2.build_decision_list(trainData, 'tweets', False, True)
    correct = 0
    total = 0
    mfs = MFS_counter(trainData, True)
    oldStats = {}
    for instance in testData['tweets']:
        result = declist2.classify(testData['tweets'][instance], old, mfs)
        answers = testData['tweets'][instance]['answers']
        if result in testData['tweets'][instance]['answers']:
            correct += 1
        total += 1
        # official score
        if result in oldStats:
            if answers[0] in oldStats[result]:
                oldStats[result][answers[0]] += 1
            else:
                oldStats[result][answers[0]] = 1
        else:
            oldStats[result] = {}
            oldStats[result][answers[0]] = 1
    accuracy = correct / total
    print("The old version achieved an accuracy of", accuracy)

    # new version
    #trainData = get_negated_dict(trainData)
    #testData = get_negated_dict(testData)
    new = declist.build_decision_list(trainData, 'tweets', False, True)
    print(new[:30])
    print(new[len(new) - 30 : ])
    newStats = {}
    for instance in testData['tweets']:
        result = declist.classify(testData['tweets'][instance], new, mfs)
        answers = testData['tweets'][instance]['answers']
        if result in answers:
            correct += 1
        total += 1
        # official score
        if result in newStats:
            if answers[0] in newStats[result]:
                newStats[result][answers[0]] += 1
            else:
                newStats[result][answers[0]] = 1
        else:
            newStats[result] = {}
            newStats[result][answers[0]] = 1
    accuracy = correct / total
    print("The new version achieved an accuracy of", accuracy)

    newStats = postprocess_stats(newStats)
    oldStats = postprocess_stats(oldStats)
    return (oldStats, newStats)

def postprocess_stats(stats):
    guesses = []
    truths = []
    for guess in stats.keys():
        guesses.append(guess)
        for truth in stats[guess].keys():
            truths.append(truth)
    guesses = set(guesses)
    truths = set(truths)
    for g in guesses:
        if g not in truths:
            for g2 in guesses:
                stats[g2][g] = 0
    for g in guesses:
        for t in truths:
            if t not in stats[g]:
                stats[g][t] = 0
    return stats


if __name__ == "__main__":
    main()
