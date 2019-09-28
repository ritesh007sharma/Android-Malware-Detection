#author Ritesh Sharma
#Implemented an ID3 algorithm, using functions such as information gain, entropy,
#Recursion.
#Machine learning Fall 2019 Assignment 1
import csv
from math import log2
import numpy as np


vals = []
save = []
def main():
    # Reading all the folds from the folder.
    with open('/home/ritesh/Desktop/cs5350/experiment-data/data/CVfolds/fold1.csv') as f:    
        r = csv.reader(f)
        first = [x for x in r]
        del first[:1]
    with open('/home/ritesh/Desktop/cs5350/experiment-data/data/CVfolds/fold2.csv') as f:    
        r = csv.reader(f)
        second = [x for x in r]
        del second[:1]
    with open('/home/ritesh/Desktop/cs5350/experiment-data/data/CVfolds/fold3.csv') as f:    
        r = csv.reader(f)
        third = [x for x in r]
        del third[:1]
    with open('/home/ritesh/Desktop/cs5350/experiment-data/data/CVfolds/fold4.csv') as f:    
        r = csv.reader(f)
        fourth = [x for x in r]
        del fourth[:1]
    with open('/home/ritesh/Desktop/cs5350/experiment-data/data/CVfolds/fold5.csv') as f:    
        r = csv.reader(f)
        fifth = [x for x in r]
        del fifth[:1]
    #created a different val to store all the folds.
    vals.append(first)
    vals.append(second)
    vals.append(third)
    vals.append(fourth)
    vals.append(fifth)
    
    folds = [first, second, third, fourth, fifth]
    valuesList, bestDepth = crossValidation(folds)
    
    #openning a training file and storing in a node. 
    with open('/home/ritesh/Downloads/experiment-data/data/train.csv') as f:    
        r = csv.reader(f)
        df = [x for x in r]
        del df[:1]
        
    node = [df,{}]   
    #calling ID3 algorithm in training set.
    rules = ID3(node, 10)
    
    #Calculation of depth.
    depth = 0
    for i,j in rules:
        depth = max(depth, len(j))
  

    with open('/home/ritesh/Downloads/experiment-data/data/test.csv') as f:       
        r = csv.reader(f)
        test_data = [x for x in r]   
        del test_data[:1]
    
    #Printing based on submission guideline.
    print("Most common label in the data with number(train): ",most_common_label(node[0]))
    print("Entropy of the data(train): ", calculate_entropy(node[0]))
    print("Best Feature and Information Gain(Number is labels[0-21]: ", info_gain(node[0]))
    print("Accuracy on the training set: ", compare(df, save))
    print("Accuracy on the test set: ", compare(test_data, save))
    print("Cross_validation accuracies for each fold: ", valuesList)
    print("Best Depth(9) Avg accuracy: ", bestDepth)
    print("Accuracy on test set using best depth: ",compare(test_data, rules))
    print("Error on test set using best depth: ",1 - compare(test_data, rules))
    print("Depth: ", depth)
    
    return 0

#ID3 class that recurses and run an algorithm.
def ID3(node, depth):
   #if depth is 0 we return .
    if depth == 0:
        return save
    
    if calculate_entropy(node[0]) == 0:
        output = [node[0][0][0],node[1]]        
        save.append(output)        
    else:
         
        #splitTree. 
         nodes = splitTree(node, info_gain(node[0]))
    
         for node in nodes:
            ID3(node, depth - 1)
         
    return save

#split tree funtion that splits the tree.
def splitTree(node, feature):
    
    data = node[0]
    nodes = []
    if feature == 0:
        return nodes
    #getting unique labels to calculate best info gain.
    for val in getUnique(get_column(data,feature)):
        store = []            
        for r in range(len(data)):
            if data[r][feature] == val:
                store.append(data[r])
                        
        dic = node[1].copy()       
        dic.update({feature:val})
        arr = [store, dic]
        nodes.append(arr)

    return nodes
         
#Funtion that returns most common label.
def most_common_label(data):
    labels = get_column(data,0)
    totalT = 0
    totalF = 0
    
    for i in range(len(labels)):
        if labels[i] == 'e':
            totalT += 1
        elif labels[i] == 'p':
            totalF += 1
    if totalT > totalF:
        return 'e ', max(totalT, totalF)
    else:
        return 'p ', max(totalT, totalF)


#funtion that calculates entropy and returns entropy.
def calculate_entropy(data):

    labels = get_column(data,0)    
    totalT = 0
    totalF = 0
    total = len(labels)
    for i in range(len(labels)):
        if labels[i] == 'e':
            totalT += 1
        elif labels[i] == 'p':
            totalF += 1
            
    if totalT == 0 or totalF == 0:
        return 0
    else:
        #print(-(totalT/total)*log2(totalT/total) - (totalF/total)*log2(totalF/total))
        return -(totalT/total)*log2(totalT/total) - (totalF/total)*log2(totalF/total)
        
        
    #Funtion that calculates information gain.
def info_gain(data): 

    entropyS = calculate_entropy(data)
    labels = get_column(data,0)
    features = extract_features(data)

    infoGain = []
    
    for col in range(len(features[0])):
        feature = get_column(features,col)
        
        store = []        
        for i in range(len(feature)):
            row = []    
            row.append(labels[i])
            row.append(feature[i])
            store.append(row)
        featureList = store    
        
        dataList = []
        expEntropy = 0

       #Calculating expected entropy.
        for val in getUnique(get_column(featureList,1)):
            storage = []            
            for r in range(len(featureList)):
                if featureList[r][1] == val:
                    storage.append(featureList[r])
            dataList.append(storage)
            
           #calculating exp entropy.
        for i in range(len(dataList)):
            if len(dataList[i]) == 0:
                print(dataList)
            expEntropy += (len(dataList[i])/len(featureList))*calculate_entropy(dataList[i])
        
       # print(exp_entropy)
       # calculating information gain by substracting expected entropy.
        IG = entropyS - expEntropy
        infoGain.append([col + 1,IG])        
        
       
        maxInfoGain = []
        #Calculating maximum information gain for root.
        for i in range(len(infoGain)):
            if len(maxInfoGain) == 0:
                maxInfoGain = infoGain[i]
            elif infoGain[i][1] > maxInfoGain[1]:                  
                maxInfoGain = infoGain[i] 
    #breaking if all IG values are zero.          
    allZeros = True
    for i in range(len(infoGain)):
        if infoGain[i][1] != 0.0:
            allZeros = False
            break
     
    if allZeros == True:
        return 0
    
    return maxInfoGain[0]
    

#Funtion that return a column.
def get_column(data,i):
    feature = []    

    for r in range(len(data)):
        feature.append(data[r][i])
    
    return feature   

#funtion that extracts feature from the data.
def extract_features(data):
    featureList = []   

    for r in range(len(data)):
        store = []

        for i in range(1, len(data[0])):
            store.append(data[r][i])

        featureList.append(store)
        
    return featureList
    
#Funtion that gets unique values from the given data set. 
def getUnique(data):
    
    dic = set()
    store = []
    for i in data:
        if i not in dic:
            store.append(i)
            dic.add(i)
    return store

#Funtion that predicts or compare two data sets based on rules. 
def compare(testData, rules):

    for r in range(len(testData)):
        for i in range(len(rules)):
            data = rules[i][1]
            keys = len(data.keys())
            c = 0
            for key, value in data.items():
                if testData[r][key] == value:
                    c += 1
                else:
                    continue
            if c == keys:
                testData[r].append(rules[i][0])
    
    match = 0
    total = len(testData)
    for row in range(len(testData)):
       if testData[row][0] == testData[row][-1]:
           match += 1
    
    for row in range(len(testData)):
        if len(testData[row]) == 23:
            testData[row].pop(22)
           
    #calculating the accuracy.
    accuracy = match / total
                    
    return accuracy  

#Function that does cross validation of the given folds. 
def crossValidation(folds):
   
    aveAccList = []
    acc = 0
    aveAcc = 0
    for i in range(len(vals)):
        test = folds.pop(i)
        folds.clear()
        for k in range(len(vals)):
            folds.append(vals[k])
        train  = []
        for j in range(len(vals)):
            if i == j:
                continue
            else:
                train.append(vals[j])
        val = []
        for df in train:
            val = val + df     
        node = [val,{}]  
        ru = ID3(node,9) 
        acc = compare(test, ru)
        aveAccList.append(acc)
        save.clear()
    
    for i in range(len(aveAccList)):
        aveAcc = aveAcc + aveAccList[i]
    aveAcc = aveAcc/len(aveAccList)
    standardD = np.std(aveAccList)
    
    return aveAccList, aveAcc
    

if __name__ == "__main__":
    main()