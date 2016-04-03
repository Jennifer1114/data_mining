#!/usr/bin/python

import sys			#used for reading arguments
import urllib2			#used for getting files directly from prof's site
from os import system		#used to call subscripts
import math			#used for gaussian function
from operator import truediv

#------------------SCRIPT FUNCTIONS--------------------

#load datasets in to memory)
#first split divides dataset, -1(class1) and +1(class2)
#second split gets attribute-value pair as dictionary
#------------------------------------------------------
def parseTrainData(trainFile):
	class1 = []
	class2 = []

	for line in open(trainFile):
		info = line.split()
		dict = {}
		for item in info[1:]:
			keyval = item.split(':')
			dict[keyval[0]]=keyval[1]
			if info[0] == '-1':
				flag = 'class1'
			if info[0] == '+1':
				flag = 'class2'
		if (flag == 'class1'):
			class1.append(sorted(dict.items()))
		else:
			class2.append(sorted(dict.items()))

	return {'-1':class1, '+1':class2}

#load test dataset to memory
#-------------------------------------------------------
def parseTestData(testFile):
	classTest = []
	trueClass = []
	
	for line in open(testFile):
		info = line.split()
		dict = {}
		for item in info[1:]:
			keyval = item.split(':')
			dict[keyval[0]]=keyval[1]
			trueClass.append(info[0])
		classTest.append(sorted(dict.items()))
	
	return classTest, trueClass

#calculate mean. for unrepresented attributes, the sum uses
#value zero to use for mean calculation
#---------------------------------------------------------
def calcMean(class1, class2, maxlen):

	sum1 = [0]*maxlen
	count1 = [0]*maxlen
	sum2 = [0]*maxlen
	count2 = [0]*maxlen

	for list in class1:
		for pair in list:
			sum1[int(pair[0])-1] += int(pair[1])
			count1[int(pair[0])-1] += 1

	for list in class2:
		for pair in list:
			sum2[int(pair[0])-1] += int(pair[1])
			count2[int(pair[0])-1] += 1
	
	mean1 = map(truediv, sum1, count1)
	mean2 = map(truediv, sum2, count2)
	return {'-1':mean1, '+1':mean2}

#calculate the standard deviation of each attribute
#--------------------------------------------------------
def calcStdev(class1, class2, maxlen, mean1, mean2):

	sqdiff1 = [0]*maxlen
	sqdiff2 = [0]*maxlen

	for list in class1:
		for pair in list:
			sqdiff1[int(pair[0])-1] += pow(float(pair[1])-float(mean1[int(pair[0])-1]),2.0)

	for list in class2:
		for pair in list:
			sqdiff2[int(pair[0])-1] += pow(float(pair[1])-float(mean2[int(pair[0])-1]),2.0)

	stdev1 = [u / float(len(class1)) for u in sqdiff1]
	stdev1 = [math.sqrt(v) for v in stdev1]
	stdev2 = [u / float(len(class2)) for u in sqdiff2]
	stdev2 = [math.sqrt(v) for v in stdev2]

	return {'-1':stdev1, '+1':stdev2}


#calculate the probability of each attribute value
#using Normal (Gaussian) distribution method
#------------------------------------------------------------
def calcProb(x, mean, stdev):
	if (stdev != 0):
		exp = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
		return (1 / (math.sqrt(2*math.pi) * stdev)) * exp
	else:
		exp = math.exp(-(math.pow(x-mean,2)/(2*math.pow(0.01,2))))
		return (1 / (math.sqrt(2*math.pi) * 0.1)) * exp


#calculate gaussian probability density - for training data
#------------------------------------------------------------
def calcGauss(testClass, maxlen, mean1, mean2, stdev1, stdev2):

	class_prob1 = []
	class_prob2 = []

	for list in testClass:
		for pair in list:
			class_prob1.append(calcProb(float(pair[1]), mean1[int(pair[0])-1], stdev1[int(pair[0])-1]))
	for list in testClass:
		for pair in list:
			class_prob2.append(calcProb(float(pair[1]), mean2[int(pair[0])-1], stdev2[int(pair[0])-1]))

	return {'-1':class_prob1, '+1':class_prob2}


#choose the highest probability and assign class
#-----------------------------------------------------------
def calcPredict(gauss):

	prediction = []

	for index in range(len(gauss['+1'])):
		first = abs(gauss['-1'][index])
		second = abs(gauss['+1'][index])		
		if (abs(first - max(first, second)) < 1e-09):
			prediction.append('-1')
		else:
			prediction.append('+1')

	return prediction	

#find the confusion matrix i.e. true/false positive/negative
#-----------------------------------------------------------
def cmpTestClass(prediction, trueClass):
	
	confMatrix = [0]*4

	for index in range(len(prediction)):
		if (prediction[index] == '-1' and trueClass[index] == '-1'):
			confMatrix[0] += 1
		if (prediction[index] == '+1' and trueClass[index] == '-1'):
			confMatrix[1] += 1
		if (prediction[index] == '-1' and trueClass[index] == '+1'):
			confMatrix[2] += 1
		if (prediction[index] == '+1' and trueClass[index] == '+1'):
			confMatrix[3] += 1

	return confMatrix


#------------------- MAIN SCRIPT --------------------------

#read in arguments and assign to train and test files
trainFile = sys.argv[1]
testFile = sys.argv[2]

#parse training data to be readable
print "Parsing training dataset file...\n"
trainData = parseTrainData(trainFile)
print "Done parsing\n"

#get max number of attributes
if (max(len(l) for l in trainData['-1']) > max(len(l) for l in trainData['+1'])):
	maxlen = max(len(l) for l in trainData['-1'])
else:
	maxlen = max(len(l) for l in trainData['+1'])

#find means and standard deviations of both classes
print "Calculating mean value of all attributes...\n"
means = calcMean(trainData['-1'], trainData['+1'], maxlen)
print "Mean calculated\n\nCalculating standard deviation of all attributes...\n"
devs = calcStdev(trainData['-1'], trainData['+1'], maxlen, means['-1'], means['+1'])
print "Standard deviation calculated\n"

#parse test data to be readable
print "Parsing test dataset file...\n"
testData,trueClass = parseTestData(testFile)
dummyVar,trainClass = parseTestData(trainFile)

print "Done parsing\n\nCalculating class probabilities...\n"
gauss = calcGauss(testData, maxlen, means['-1'], means['+1'], devs['-1'], devs['+1'])

print "Done calculating probabilities\n\nDetermining class predictions...\n"
prediction = calcPredict(gauss)

print "Done predicting classes\n\nBeginning accuracy measurements...\n"
confMatrix = cmpTestClass(prediction, trueClass)
confMatrix1 = cmpTestClass(prediction, trainClass)
print "Done calculating Classifier quality\n\nResults...\n"

print ' '.join(map(str,confMatrix))
print ' '.join(map(str,confMatrix1))
