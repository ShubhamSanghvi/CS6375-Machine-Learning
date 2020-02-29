# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:03:02 2020

@author: shubh
"""

import numpy as np
import pandas as pd
import math

def getEntropy(entropy_df):
    # assuming zeroth column has the variable
    if entropy_df.shape[1] == 0:
        #print("ERROR: Entropy function")
        exit()
    
    
    counts  = entropy_df.iloc[:,0].value_counts()
    probs = counts/sum(counts)
    
    #print(probs)
    
    h = 0
    for p in probs:
        h += p * math.log(p,2)
    
    return -h

# question == column for the sake of this dataset
def getEntropy_question(q_entropy_df,ques):
    total_h = 0
    total_samples = q_entropy_df.shape[0]
    weights = q_entropy_df[ques].value_counts()/total_samples
    
    for split_label in q_entropy_df[ques].unique():
        # split data for this label
        df_split = q_entropy_df[q_entropy_df[ques] ==  split_label]
        total_h += weights[split_label] * getEntropy(df_split)
        #print(total_h)
    
    return total_h
    
class treeNode:
    def __init__(self):
        # intialize a node here
        self.ntype = 'internal'
        self.result = None
        self.question = None
        self.infoGain = None
        self.children = {}


def build_tree(data,entropy,parent_maxvote,level):
    node = treeNode()
    if data.shape[0] == 0:
        # that means we dont have any samples
        node.ntype = 'leaf'
        node.result = parent_maxvote
    elif len(train[0].unique()) == 1:
        # we just have one label
        node.ntype = 'leaf'
        node.result = data[0].mode()
    elif data.shape[1] == 1:
        # we have no more columns to split on
        node.ntype = 'leaf'
        node.result = data[0].mode()
    else:
        min_entropy = entropy
        new_question = None
        for column in data:
            if column == 0:
                continue
            label_entropy = getEntropy_question(data,column)
            #print("Col : ",column,"\tentropy :",label_entropy)
            
            # this will take care of ties. We are going from left to right
            if label_entropy  < min_entropy:
                min_entropy = label_entropy 
                new_question = column
            
        if new_question is not None:
            print("level: ",level," split > ",new_question)
            node.question = new_question
            node.infoGain = entropy - min_entropy
#            node.children = [None for _ in data[new_question].unique()]


            node.result = data[0].mode()
            new_data = data.drop(new_question,1)
            
            print("Counts for this label:",new_question)
            print(data[new_question].value_counts())
            
            for split_label in data[new_question].unique():
                print("----trying: ",new_question,',',split_label)
                # split data for this label
                df_split = new_data[data[new_question]==  split_label]

                parent_entropy = getEntropy(df_split)

                child = build_tree(df_split,parent_entropy ,node.result,level+1)
                if child.ntype == 'leaf':
                    print("-----Leaf - ",split_label)

                node.children[split_label]= child
        else:
            node.ntype = 'leaf'
            node.result = data[0].mode()

    return node


def print_tree(node,level):
    if not node:
        return

    print("\t"*level,node.ntype)
    if node.ntype == 'leaf':
        print("\t"*level,"Node.label:",node.result[0])
    else:
        print("\t"*level,"Node.I:",node.infoGain)
        print("\t"*level,"Node.question,",node.question)
        for child in node.children:
            print("\t"*(level+1),str(child))
            print_tree(node.children[child],level+1)


def make_predictions(data,node):

    if node.ntype == 'leaf':
        return node.result
    
    col = node.question
    # see what label do we have
    if data[col] in node.children:
        child = node.children[data[col]]
        return make_predictions(data,child)
    else:
        return node.result
    
    
def check(X):
    if X.iloc[0]==X.iloc[1]:
        return 1
    else:
        return 0

    

# Driver Code
if __name__ == '__main__':
    
    # read training data
    
    train = pd.read_csv("mush_train.data",header = None)
    _train = train.copy()

    train.head()
    
    unq_labels = [train[column].unique() for column in train]
    
    lmap = []
    # for each column
    # transform for ease of use
    for column in train:
        col_map = {}
        idx = 0
        for label in train[column].unique():
            col_map[label] = idx
            idx = idx + 1
        
        lmap.append(col_map)
        train[column] = train[column].map(col_map)

    max_entropy = getEntropy(train)
    max_votes = train[0].mode()
    
    tree = build_tree(train,max_entropy,max_votes,0)
    print_tree(tree,0)
    
    predictions = train.apply(make_predictions,axis=1,args=(tree,))

    train = pd.concat([predictions,train],axis=1)
    results = train.apply(check,axis=1)
    train_accuracy = sum(results)/len(results)
    
    test = pd.read_csv("mush_test.data",header = None)
    
    for column in test:
        test[column] = test[column].map(lmap[column])
    
    test_pred = test.apply(make_predictions,axis=1,args=(tree,))

    test_res = pd.concat([test_pred,test],axis=1)
    final_results = test_res.apply(check,axis=1)
    test_accuracy = sum(final_results )/len(final_results )
    
    
    
    
    
    
    
