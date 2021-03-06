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
import sys
from parseTweet import *
from crossval import *
from helpers import *
from scorer import *
from MegaClassifier import *

def main():
    try:
        k = int(sys.argv[2])
        flag = True
    except ValueError:
        flag = False

    trainData = parse_tweets(sys.argv[1], 'B')
    mega = MegaClassifier()
    if flag:
        chunks = k_chunks(trainData, k)
        crossvalTest(chunks, mega)
    else:
        testVsTrain(trainData, sys.argv[2], mega)

def debugNBC(chunks, classifier):
    testData = chunks[0]
    trainData = combine_chunks(chunks[1:])
    classifier.setTrainData(trainData)
    classifier.setTestData(testData)
    classifier.buildNaiveBayesClassifier()
    classifier.debugNBC()

def crossvalTest(chunks, classifier):
    scores = []
    for i in range(len(chunks)):
        testData = chunks[i]
        trainSet = [chunks[j] for j in range(len(chunks)) if j != i]
        trainData = combine_chunks(trainSet)
        stats = test(trainData, testData, classifier)
        newScore = scorer(stats)
        scores.append(newScore)

    print("In cross-validation, the chunks achieved the following scores")
    for i in range(len(scores)):
        print("Chunk", i, ":", scores[i])
    print("Average score:", sum(scores)/len(scores))



def combine_chunks(chunks):
    combined = {}
    combined['tweets'] = chunks[0]['tweets'].copy()
    for i in range(1, len(chunks)):
        combined['tweets'].update(chunks[i]['tweets'])
    return combined

def testVsTrain(trainData, filename, classifier, showScore=False):
    testData = parse_tweets(filename, 'B')
    stats = test(trainData, testData, classifier)
    if showScore:
        newScore = scorer(stats)
        print("The new version got an official score of", newScore)


def test(trainData, testData, classifier):
    mfs = MFS_counter(trainData, True)

    classifier.setTrainData(trainData)
    classifier.setTestData(testData)
    classifier.buildDecisionList()
    classifier.buildNaiveBayesClassifier()
    classifier.buildCountSVM()
    classifier.buildKNearestNeighbors()

    newStats = {}
    for instance in testData['tweets']:
        result = classifier.classifyInstanceByCountSVM(testData['tweets'][instance])
        answers = testData['tweets'][instance]['answers']
        print(instance, '\t', result)
        """
        if result in answers:
            correct += 1
        total += 1
        """
        # official score
        if result in newStats:
            if answers[0] in newStats[result]:
                newStats[result][answers[0]] += 1
            else:
                newStats[result][answers[0]] = 1
        else:
            newStats[result] = {}
            newStats[result][answers[0]] = 1

    newStats = postprocess_stats(newStats)
    return newStats

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
