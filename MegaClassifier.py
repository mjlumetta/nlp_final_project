"""
A class that ties together our classifying stuff.
"""
from declist import *
from helpers import *
from naivebayes import *
from sklearn import svm, feature_extraction, neighbors
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
        self.CountSVM = svm.LinearSVC()
        self.classes = []
        self.cVectorizer = None
        self.KNN = neighbors.KNeighborsClassifier(weights='distance')
        # self.algorithmScores = dict(naiveBayes=34.68, svm=54.34, knn=29.58, decisionList=46.02)
        self.algorithmScores = dict(naiveBayes=34.68, svm=54.34, decisionList=46.02)
        self.weights = self.getWeights()

    def getWeights(self):
        total = sum(self.algorithmScores.values())
        weights = {}
        for k, v in self.algorithmScores.items():
            weights[k] = v / total
        return weights

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
        self.classes = self.getClasses()
        self.cVectorizer = feature_extraction.text.CountVectorizer()
        self.cVectorizer.fit(self.getDocuments())

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

    def classifyInstanceMega(self, instance):
        naiveBayesResult = self.classifyInstanceByNaiveBayes(instance)
        decisionListResult = self.classifyInstanceByDecisionList(instance)
        #knnResult = self.classifyInstanceByKNN(instance)
        svmResult = self.classifyInstanceByCountSVM(instance)
        neutralList = ['neutral', 'objective', 'objective-or-neutral']
        results = {}
        for c in self.classes:
            results[c] = 0.0
        results[naiveBayesResult] += self.weights['naiveBayes']
        # results[knnResult] += self.weights['knn']
        results[svmResult] += self.weights['svm']
        results[decisionListResult] += self.weights['decisionList']
        maxScore = 0.0
        maxClass = []
        for k, v in results.items():
            if v >= maxScore:
                maxScore = v
                maxClass.append(k)
        if len(maxClass) == 1:
            return maxClass[0]
        elif svmResult in maxClass:
            return svmResult
        else:
            return decisionListResult

    def classifyInstance(self, instance):
        svmResult = self.classifyInstanceByCountSVM(instance)
        decisionListResult = self.classifyInstanceByCountSVM(instance)
        neutralList = ['neutral', 'objective', 'objective-or-neutral']
        if svmResult == decisionListResult:
            return svmResult
        elif svmResult not in neutralList:
            # decision list wins tiebreak if one is pos and other is neg
            return svmResult
        else:
            return decisionListResult

    def classifyInstanceByCountSVM(self, instance):
        featureVector = self.cVectorizer.transform([' '.join(instance['words'])])
        result = self.CountSVM.predict(featureVector)[0]
        return self.classes[result]

    def classifyInstanceByKNN(self, instance):
        featureVector = self.cVectorizer.transform([' '.join(instance['words'])])
        result = self.KNN.predict(featureVector)[0]
        return self.classes[result]

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

    def buildCountSVMByHand(self):
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
                    # print("Answer index is", i)
            classifications.append(answerIndex)
        self.CountSVM.fit(numpy.array(instanceVectors), numpy.array(classifications))

    def buildCountSVM(self):
        documents = self.getDocuments()
        featureVectors = self.cVectorizer.transform(documents)
        classifications = self.getClassifications()
        self.CountSVM.fit(featureVectors, classifications)

    def buildKNearestNeighbors(self, k=5, weighted=True):
        if weighted:
            weightParam = 'distance'
        else:
            weightParam = 'uniform'
        self.KNN.set_params(weights=weightParam, n_neighbors=k)
        featureVectors = self.cVectorizer.transform(self.getDocuments())
        classifications = self.getClassifications()
        self.KNN.fit(featureVectors, classifications)


    def getDocuments(self):
        documents = []
        for k in self.trainData['tweets']:
            tweet = self.trainData['tweets'][k]
            documents.append(' '.join(tweet['words']))
        return documents

    def getClassifications(self):
        classifications = []
        for k in self.trainData['tweets']:
            tweet = self.trainData['tweets'][k]
            answerIndex = -1
            for i in range(len(self.classes)):
                if tweet['answers'][0] == self.classes[i]:
                    answerIndex = i
                    # print("Answer index is", i)
            classifications.append(answerIndex)
        return numpy.array(classifications)

    """
    getFeatureMap
    Builds a dictionary that maps each feature to its index in
    a feature vector. For use with SVM.
    """
    def getFeatureMap(self):
        features = set()
        for instance in self.trainData['tweets']:
            instanceFeatures = set(get_bag_of_words(self.trainData['tweets'][instance]))
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
        features = get_bag_of_words(instance)
        for f in features:
            if f in featureMap:
                vector[featureMap[f]] += 1
        return vector

    def buildFeaturePresenceVector(self, instance, featureMap):
        vector = []
        for i in range(len(featureMap)):
            vector.append(0)
        features = get_bag_of_words(instance)
        for f in features:
            if f in featureMap:
                vector[featureMap[f]] = 1
        return vector


        
