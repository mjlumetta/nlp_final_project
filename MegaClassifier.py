"""
A class that ties together our classifying stuff.
"""
from declist import *
from helpers import *

class MegaClassifier:
    casefold = True
    negate = True
    stopwords = False
    decisionList = []

    def __init__(self):
        self.trainData = {}
        self.testData = {}

    def setTrainData(self, newData):
        if self.negate:
            self.trainData = get_negated_dict(newData)
        else:
            self.trainData = newData

    def setTestData(self, newData):
        if self.negate:
            self.testData = get_negated_dict(newData)
        else:
            self.testData = newData

    def getTrainData(self):
        return self.trainData

    def getTestData(self):
        return self.testData

    def buildDecisionList(self):
        self.decisionList = build_decision_list(self.trainData, "tweets", \
                self.stopwords, self.casefold)

    def setStopwords(new):
        self.stopwords = new

    def setCasefold(new):
        self.casefold = new

    def setNegate(new):
        self.negate = new
