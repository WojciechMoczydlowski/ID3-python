import csv
import pandas as pd
import numpy as np
import random
from pprint import pprint

dataset = pd.read_csv(r"C:\Users\German\Desktop\Machine Learning\Assignment 1\mushrooms.csv",delimiter=',')

#shape[0] is refering to the total number of rows
#dataset['id'] = [random.randint(0,1000) for x in range(dataset.shape[0])]

dataset = dataset.astype(str)

#CALCULATING ORIGINAL ENTROPY
target_attribute = dataset['class']
def entropy(target_attribute="class"):
    elements,occurrences = np.unique(target_attribute,return_counts = True)
    for i in range(len(elements)):
        entropy_value = np.sum([(-occurrences[i]/np.sum(occurrences))*np.log2(occurrences[i]/np.sum(occurrences))])
        return(entropy_value)

#CALCULATING INFORMATION GAIN
def information_gain(dataframe, attr, target_attribute="class"):
    #Calculate the entropy of the total dataset
    total_entropy = entropy(dataframe[target_attribute])
    vals,occurrences= np.unique(dataset[attr],return_counts=True)
    #Weighted Entropy
    for i in range(len(vals)):
        weighted_entropy = np.sum([(occurrences[i]/np.sum(occurrences))*entropy(dataset.where(dataset[attr]==vals[i]))])
    #Calculate the information gain
    information_gain = total_entropy - weighted_entropy
    return information_gain

def main():
    
    the_entropy = entropy(target_attribute)
    print(the_entropy)
    
    all_attributes_IG = dataset[['cap-shape','cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'veil-color', 'spore-print-color', 'population', 'habitat']]
    all_gains = [] 
    max_gain = -99999
    
    for i in all_attributes_IG:
        each_information_gain = information_gain(dataset, all_attributes_IG)
        print(the_information_gain)
        all_gains.append(each_information_gain)
    
    for j in range(len(all_gains)):
        if all_gains[j] > max_gain:
            max_gain = all_gains[j]
    print(max_gain)

    
    #print(list(dataset.columns.values))
    #print(dataset.head())
    #print(dataset.head())

if __name__ == '__main__':
   main()





