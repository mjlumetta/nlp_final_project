"""
A class that ties together our classifying stuff.
"""
from declist import *
from helpers import *
from naivebayes import *
from sklearn import svm
import numpy

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
        self.SVM = svm.SVC(cache_size=1000)
        self.classes = []

    def getClasses(self):
        classes = set()
        for k in self.trainData['tweets']:
            instance = self.trainData['tweets'][k]
            for a in instance['answers']:
                classes.add(a)
        return sorted(list(classes))

    def setTrainData(self, newData):
        if self.negate:
            self.trainData = get_negated_dict(newData)
        else:
            self.trainData = newData
        self.mfs = MFS_counter(self.trainData)
        self.getClasses()

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
        features = get_features(instance, range(2,4), self.casefold, [], 3)
        return self.classifyInstanceByDecisionListKHelper(instance, k, features)

    def classifyInstanceByDecisionListLess(self, instance):
        features = get_bag_of_words(instance, [], self.casefold)
        for entry in decision_list:
            if entry[1] in features:
                return entry[0]
        return MFS

    def classifyInstanceByDecisionListLessK(self, instance, k):
        features = get_bag_of_words(instance, [], self.casefold)
        return self.classifyInstanceByDecisionListKHelper(instance, k, features)

    def classifyInstanceByDecisionListKHelper(self, instance, k, features):
        if k > len(features):
            k = len(features)
        sigFeatures = []
        for entry in self.decisionList:
            if entry[1] in features:
                sigFeatures.append(entry)
                if len(sigFeatures) == k:
                    break
        counts = {}
        for entry in sigFeatures:
            if entry[0] not in counts:
                counts[entry[0]] = 1
            else:
                counts[entry[0]] += 1
        maxCount = 0
        maxSense = []
        for sense in counts:
            if counts[sense] >= maxCount:
                maxCount = counts[sense]
                maxSense.append(sense)
        if len(maxSense) == 1:
            return maxSense[0]
        else:
            for entry in sigFeatures:
                if entry[0] in maxSense:
                    return entry[0]
        return self.mfs


    def debugNBC(self):
        print("Sentiment probabilities")
        print(self.naiveBayesClassifier[0])
        print("Feature probabilities")
        positive = self.naiveBayesClassifier[1]['positive']
        print(list(positive.items())[:50])

    def buildCountSVM(self):
        featureMap = self.getFeatureMap()
        print(len(featureMap))
        instanceVectors = []
        classifications = []
        for key in self.trainData['tweets']:
            instance = self.trainData['tweets'][key]
            instanceVectors.append(self.buildFeatureCountVector(instance, featureMap))
            answerIndex = -1
            for i in range(len(self.classes)):
                if instance['answers'][0] == self.classes[i]:
                    answerIndex = i
            classifications.append(answerIndex)
        # self.SVM.fit(numpy.array(instanceVectors), numpy.array(classifications))

    """
    getFeatureMap
    Builds a dictionary that maps each feature to its index in
    a feature vector. For use with SVM.
    """
    def getFeatureMap(self):
        features = set()
        for instance in self.trainData['tweets']:
            instanceFeatures = set(get_features(self.trainData['tweets'][instance]))
            features.update(instanceFeatures)
        order = sorted(list(features))
        featureMap = {}
        for i in range(len(order)):
            featureMap[order[i]] = i
        return featureMap

    def buildFeatureCountVector(self, instance, featureMap):
        vector = []
        for i in range(len(featureMap)):
            vector.append(0)
        features = get_features(instance)
        for f in features:
            if f in featureMap:
                vector[featureMap[f]] += 1
        return vector

    def buildFeaturePresenceVector(self, instance, featureMap):
        vector = []
        for i in range(len(featureMap)):
            vector.append(0)
        features = get_features(instance)
        for f in features:
            if f in featureMap:
                vector[featureMap[f]] = 1
        return vector


        
