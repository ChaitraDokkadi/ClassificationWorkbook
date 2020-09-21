#!/usr/bin/env python
# coding: utf-8

# In[23]:


import sys
import os
import numpy as np
from math import sqrt
import random
import pandas as pd

def gini_index(samples, classes):
    gini = 0.0
    samples_count = float(sum([len(group) for group in samples]))
    
    for group in samples:
        if float(len(group)) < 1.0:
            continue
        total_score = 0.0
        for current_class in classes:
            current_class_score = [row[-1] for row in group].count(current_class) / float(len(group))
            total_score += current_class_score**2
        gini += (1.0 - total_score) * (float(len(group)) / samples_count)
    return gini

def evaluate_best_split(data):
    class_values = np.unique(data[:,-1])
    
    best_index = 10000
    best_value = 10000
    best_score = 10000
    best_groups = None
    
    for index in range(len(data[0])-1):
        for row in data:
            left = []
            right = []
            for item in data:
                if type(row[index]) is str:
                    left.append(item) if item[index] == row[index] else right.append(item)
                else:
                    left.append(item) if item[index] <= row[index] else right.append(item)
            ordered_groups = np.array(left), np.array(right)
            
            gini = gini_index(ordered_groups, class_values)
            if gini < best_score:
                best_index = index
                best_value = row[index]
                best_score = gini
                best_groups = ordered_groups
    node = {}
    node['index'] = best_index
    node['value'] = best_value
    node['left'], node['right'] = best_groups
    return node

def splitting(node,depth):
    left = node['left']
    right = node['right']
    if left.size==0 or right.size==0:
        classes = left[:,-1] if right.size == 0 else right[:,-1]
    
        node['left'] = max(classes, key = list(classes).count)
        node['right'] = node['left']
        return

    if depth >= maximum_depth:
        classes = left[:,-1]
        node['left'] = max(classes, key = list(classes).count)
        classes = right[:,-1]
        node['right'] = max(classes, key = list(classes).count)
        return
    
    if len(left) > 1:
        node['left'] = evaluate_best_split(left)
        splitting(node['left'],depth+1)
    else:
        classes = left[:,-1]
        node['left'] = max(classes, key = list(classes).count)
    
    if len(right) > 1:
        node['right'] = evaluate_best_split(right)
        splitting(node['right'],depth+1)
    else:    
        classes = right[:,-1]
        node['right'] = max(classes, key = list(classes).count)

def get_decision_tree(train_data,depth):
    root = evaluate_best_split(train_data)
    splitting(root,depth)
    return root

def get_label(row,node):
    if row[node['index']] >= node['value']:
        return get_label(row,node['right']) if type(node['right']) is dict else node['right']
    else:
        return get_label(row,node['left']) if type(node['left']) is dict else node['left']
        
def get_labels(test_data,node):
    labels = []
    for row in test_data:
        label = get_label(row,node)
        labels.append(label)
    return labels

def get_labels_ada_boost(test_data,trees,alphas):
    x = np.zeros(len(test_data))
    for (m, a) in zip(trees, alphas):
        print(a)
        labels = get_labels(test_data,m)
        updated_labels = []
        for label in labels:
            y = -1 if label == 0 else 1
            updated_labels.append(y)
        x += a * np.array(updated_labels) 
    return np.sign(x)

def ada_Boost(train_data,m):
    trees = []
    alphas = []
    
    weights = np.ones(len(train_data)) / len(train_data)
    
    while len(trees) < m:
#         print(len(trees))
        N = len(weights)
        
        new_train_data_indices = np.random.choice(N, size=N, replace=True, p=weights)
        
        new_train_data = train_data[new_train_data_indices]

        tree = get_decision_tree(new_train_data,0)
        predicted_labels = get_labels(train_data,tree)
        
        error = weights.dot(predicted_labels != train_data[:,-1])
        
        if error > 0.5:
            continue
            
        alpha = 0.5 * (np.log((1 - error)/error))
#         print(len(predicted_labels))
#         print(len(train_data[:,-1]))
        train_labels = train_data[:,-1]
        
        change = []
        for index in range(len(train_labels)):
            change.append(1 if train_labels[index] == predicted_labels[index] else -1)
        change = np.array(change)  

        change = np.asarray(change)
        weights *= np.exp(-alpha*change)
        weights /= sum(weights)

        trees.append(tree)
        alphas.append(alpha)
    
    return trees,alphas

train_label="train_label.csv"
train_label= np.loadtxt(train_label, delimiter=",",dtype='str')
train_label=np.delete(train_label,0,1)
train_label=np.delete(train_label,0,0)
train_label=train_label.astype(np.float)

file_name="train_features.csv"
data = np.loadtxt(file_name, delimiter=",",dtype='str')
data =np.asarray(data)
train_data=data.astype(np.float)
train_data=np.delete(train_data,0,1)
train_data = np.append(train_data,train_label,axis=1)

test_features="test_features.csv"
test_data= np.loadtxt(test_features, delimiter=",",dtype='str')
test_data=test_data.astype(np.float)
x = np.reshape(test_data[:,0],(len(test_data),1))
test_data=np.delete(test_data,0,1)

M = [12]
K=1
maximum_depth = 4

for m in M:
    for i in range(K):
#         print(len(train_data))
#         print(len(test_data))
        
        trees,alphas = ada_Boost(train_data,m)
        
        predicted_labels = get_labels_ada_boost(test_data,trees,alphas)
        
        predicted_labels[predicted_labels == -1] = 0
        predicted_labels[predicted_labels == 0] = '0'
        predicted_labels[predicted_labels == 1] == '1'
        print(predicted_labels)

y = np.reshape(predicted_labels, len(predicted_labels),1)
# print(predicted_labels)

y = np.reshape(predicted_labels, (len(predicted_labels),1))
# print(len(x))
# print(len(y))
z = np.append(x,y,axis=1)
z = pd.DataFrame({'id': z[:, 0], 'label': z[:, 1]})
# z.astype(int)
z.to_csv("submission.csv",index=False)


# In[ ]:




