README

Authors: Mike Lumetta, Chris Miller

There are four text files included in this directory. Three of them are 
examples of our system's output when run on test data:
	- result_labels_mega.txt (classifyInstanceMega)
	- result_labels_default.txt (classifyInstance)
	- result_labels_svm.txt (classifyInstanceByCountSVM)

The names in parentheses are the functions in MegaClassifier that 
were responsible for each result. All of these use the scikit-learn 
SVM to a large extent, with information also incorporated from other
classifiers.

The other text files, results.txt, was more for self-reference
but can be used to understand our system and approach. Much of the 
information in this file is incorporated in our paper.

The bulk of our code is run through MegaClassifier.