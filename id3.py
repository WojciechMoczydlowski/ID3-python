#import csv
import pandas as pd
import numpy as np
#from pprint import pprint

# Please change the file path when using this code
dataset = pd.read_csv(r"/home/transmatter/folders/code/uni-bam/" \
                      r"KogSys-ML/ID3-python/zoo2.csv",
                      delimiter=',',
                      names=['SKIN','HAIR','TARGET'])

#CALCULATING ORIGINAL ENTROPY
target_attribute = dataset['TARGET']
def entropy(target_attribute):
    elements,occurrences = np.unique(target_attribute,return_counts = True)
    for i in range(len(elements)):
        entropy_value = np.sum([(-occurrences[i]/np.sum(occurrences))*
                                np.log2(occurrences[i]/np.sum(occurrences))])
        return(entropy_value)

#CALCULATING INFORMATION GAIN

data2 = dataset[['SKIN', 'HAIR']]


def information_gain(dataframe, attr="HAIR", target_attribute="TARGET"):
    #Calculate the entropy of the total dataset
    total_entropy = entropy(dataframe[target_attribute])
    vals,occurrences= np.unique(dataset[attr],return_counts=True)
    #Weighted Entropy
    for i in range(len(vals)):
        weighted_entropy = np.sum([(occurrences[i]/np.sum(occurrences))*
                                   entropy(dataset.where(dataset[attr]==
                                                         vals[i]).dropna()[target_attribute])])
    #Calculate the information gain
    information_gain = total_entropy - weighted_entropy
    return information_gain


def main():
    the_entropy = entropy(dataset['TARGET'])
    the_information_gain = information_gain(dataset)
    print(the_entropy)
    print(the_information_gain)
    print(list(dataset.columns.values))
    print(dataset.head())
    
if __name__ == '__main__':
   main()







#for i in data:
    #elements, counts = np.unique(data[i],return_counts = True)
    #print(elements)
    #print(counts)

#for index in elements:
    #child_entropy = ((-counts[index]/np.sum(counts))*np.log2(counts[index]/np.sum(counts)))
    #print(child_entropy)

#for j in range(0,1):
    #weighted_entropy = np.sum([(counts[j]/np.sum(counts))*child_entropy])
    #print(weighted_entropy)

    

#information_gain = 0
#information_gain = total_entropy - weighted_entropy
#print(information_gain)








#total_weighted_entropy += weighted_entropy

#weighted_entropy = np.sum([(counts[index]/np.sum(counts))*total_child_entropy])



#dataset=dataset.drop('animal_name',axis=1)

#print(dataset.head)


##unique_elements = target_attribute.unique()
    #print(unique_elements)

    #for column in dataset:
        #for index, row in dataset[column].iteritems():
    #elements,counts = np.unique(target_attribute,return_counts = True)
#for i, row in target_attribute.iteritems():
#def entropy(target_col):