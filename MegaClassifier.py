"""
A class that ties together our classifying stuff.
"""
from declist import *
from helpers import *
from naivebayes import *

class MegaClassifier:
    casefold = True
    negate = True
    stopwords = False
    mfs = ""

    def __init__(self):
        self.trainData = {}
        self.testData = {}
        self.decisionList = []
        self.naiveBayesClassifier = ()

    def setTrainData(self, newData):
        if self.negate:
            self.trainData = get_negated_dict(newData)
        else:
            self.trainData = newData
        self.mfs = MFS_counter(self.trainData)

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

    def buildNaiveBayesClassifier(self):
        self.naiveBayesClassifier = build_classifier(self.trainData, self.stopwords, self.casefold)

    def setStopwords(new):
        self.stopwords = new

    def setCasefold(new):
        self.casefold = new

    def setNegate(new):
        self.negate = new

    def classifyInstance(self, instance):
        naiveBayesResult = self.classifyInstanceByNaiveBayes(instance)
        decisionListResult = self.classifyInstanceByDecisionList(instance)
        neutralList = ['neutral', 'objective', 'objective-or-neutral']
        if naiveBayesResult == decisionListResult:
            return naiveBayesResult
        elif decisionListResult not in neutralList:
            # decision list wins tiebreak if one is pos and other is neg
            return decisionListResult
        else:
            return naiveBayesResult

    def classifyInstanceByDecisionList(self, instance):
        return classify(instance, self.decisionList, self.mfs)

    def classifyInstanceByNaiveBayes(self, instance):
        return classifyNBC(instance, self.naiveBayesClassifier)

    def classifyInstanceByDecisionListK(self, instance, k):
        return classify(instance, self.decisionList, self.mfs)

    def debugNBC(self):
        print("Sentiment probabilities")
        print(self.naiveBayesClassifier[0])
        print("Feature probabilities")
        positive = self.naiveBayesClassifier[1]['positive']
        print(list(positive.items())[:50])

        
