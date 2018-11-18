import pandas as pd
import numpy as np

dataset = pd.read_csv(r"mushrooms.csv",delimiter=',')
target_arrray = dataset['class']

def calc_entropy(dataset):
    target_array = dataset["class"]
    elements,occurrences = np.unique(target_array,return_counts = True)
    for i in range(len(elements)):
        entropy = np.sum([(-occurrences[i]/np.sum(occurrences))*
                                np.log2(occurrences[i]/np.sum(occurrences))])
        return(entropy)

# Desc_attributes are those which are describe the instance,
# as opposed to the target attribute which classifies the result.
desc_attributes = dataset.drop(columns="class")

def calc_IG(dataset, desc_attributes, target_arrray):
    child_vals = []
    weighted_vals = []
    all_gains = []
    for index in desc_attributes:
        values,counts = np.unique(desc_attributes[index],
                                  return_counts = True)
        for j in range(len(values)):
            child_entropy = np.sum([(-counts[j]/np.sum(counts)*
                                     np.log2(counts[j]/np.sum(counts)))])
        
        child_vals.append(child_entropy)
        for foo in range(len(child_vals)):
            weighted_entropy = np.sum(([(counts[j]/np.sum(counts))*
                                        child_vals[foo]]))
        weighted_vals.append(weighted_entropy)   
        information_gain = calc_entropy(dataset) - weighted_vals[foo]
        all_gains.append(information_gain)
    all_gains = np.around(all_gains, decimals=4, out=None)
    return all_gains

def main():
    gains = calc_IG(dataset, desc_attributes, target_arrray)
#    max_gain = np.amax(all_gains)
    print("Array of all gains: ",gains)

if __name__ == '__main__':
    main()