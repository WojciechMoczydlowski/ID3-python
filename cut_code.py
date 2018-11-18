# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------

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
    
