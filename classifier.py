__author__ = 'akshaykulkarni'

import csv
import random
import numpy
import math

training        = []
testing         = []
partition_size  = 0.7

def get_accuracy(predictions,testing):
    correct             = 0
    for i in range(len(testing)):
        if testing[i][-1] == predictions[i]:
            correct     = correct +1

    return (correct*100.0)/(len(predictions)*1.0)

def get_predictions(summary,testing):
    predictions = []
    for i in range(len(testing)):
        predictions.append(get_correct_label(summary,testing[i]))

    return predictions

def get_correct_label(summary,input):
    classProability     = calculate_class_proability(summary,input)
    bestLabel, bestProb = None, -1

    for k,v in classProability.iteritems():
        if bestLabel is None or v > bestProb:
            bestProb    = v
            bestLabel   = k

    return bestLabel

def calculate_probability(inp,mean,std):
    exp = math.exp(-(math.pow(inp-mean,2)/(2*math.pow(std,2))))
    return (1 / (math.sqrt(2*math.pi) * std)) * exp

def calculate_class_proability(summary,input_vector):
    classProability     = {}
    for k,v in summary.iteritems():
        classProability[k]  = 1.0
        for i in range(len(v)):
            mean,std    = v[i]
            classProability[k]  = classProability[k] * calculate_probability(input_vector[i],std,mean)
    return classProability

def seperate_by_class(training):
    res         = {}
    for ex in training:
        if ex[-1] not in res:
            res[ex[-1]] = []
        res[ex[-1]].append(ex)

    return res

def summarize(input):
    summary = [(numpy.mean(attribute),numpy.std(attribute)) for attribute in zip(*input)]
    del summary[-1]
    return summary

def get_summary(input):
    summary	= {}
    for k,v in input.iteritems():
		summary[k]	= summarize(v)
    return summary


def load_csv(filename):
    with open(filename) as file_obj:
        reader  = csv.reader(file_obj)
        for line in reader:
            training.append([float(x) for x in line])


load_csv('pima-indians-diabetes.csv')
random.shuffle(training)
partition_idx   = int(partition_size * len(training))
testing         = training[partition_idx:]
training        = training[:partition_idx]

data_by_class   = seperate_by_class(training)
summary_by_class= get_summary(data_by_class)
predictions     = get_predictions(summary_by_class,testing)
accuracy        = get_accuracy(predictions,testing)
print accuracy