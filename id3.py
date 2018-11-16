import csv
import pandas as pd
import numpy as np
import random
from pprint import pprint

dataset = pd.read_csv(r"C:\Users\German\Desktop\Machine Learning\Assignment 1\ID3-python\mushrooms.csv",delimiter=',')

#CALCULATING ORIGINAL ENTROPY
target_attribute = dataset['class']
def entropy(target_attribute="class"):
    elements,occurrences = np.unique(target_attribute,return_counts = True)
    for i in range(len(elements)):
        entropy_value = np.sum([(-occurrences[i]/np.sum(occurrences))*np.log2(occurrences[i]/np.sum(occurrences))])
        return(entropy_value)

#CALCULATING INFORMATION GAIN
all_attributes_IG = dataset.drop(columns="class")
def informationGain(dataset,all_attributes_IG,target_attribute):
    child_vals = []
    weighted_vals = []
    all_gains = []
    for index in all_attributes_IG:
        values,counts = np.unique(all_attributes_IG[index], return_counts = True)
        for j in range(len(values)):
            child_entropy = np.sum([(-counts[j]/np.sum(counts)*np.log2(counts[j]/np.sum(counts)))])
        child_vals.append(child_entropy)
        for foo in range(len(child_vals)):
            weighted_entropy = np.sum(([(counts[j]/np.sum(counts))*child_vals[foo]]))
        weighted_vals.append(weighted_entropy)   
        information_gain = entropy(dataset) - weighted_vals[foo]
        all_gains.append(information_gain)
    all_gains = np.around(all_gains, decimals=4, out=None)
    #print(all_gains)
    #(Problem: no such thing as negative information gain)
    #BEST CLASSIFIER 
    max_gain = np.amax(all_gains)
    return max_gain 
    

def main():
    
    best_classifier = informationGain(dataset, all_attributes_IG, target_attribute)
    print(best_classifier, "is the best classifier which belongs to attribute: stalk-color-above-ring")

if __name__ == '__main__':
    main()

#ID3 IMPLEMENTATION

def id3Algorithm(d, attributes, target_attribute):
#If all examples are positive, Return single-node tree Root, with label = +
#If all examples are negative, Return single-node tree Root, with label = −
#If all target_values have the same value, return this value i.e either e or p respectively for + and -
    if len(np.unique(dataset[target_attribute])) <= 1:
        return np.unique(dataset[target_attribute])[0]
#If Attributes is empty, Return single-node tree Root, with label = most common value of Target attribute in Examples
    elif len(dataset)==0:
        elements,counts =  np.unique(target_attribute, return_counts = True)
        return np.amax(counts)[1]