#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
import numpy as np
from math import sqrt
import statistics

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
    node['gini'] = best_score
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

def print_tree(node, depth, direction):
    if type(node) is dict:
        print(depth*' ', 'X', (node['index']+1),'<=',node['value'], direction, node['gini'])
        print_tree(node['left'], depth+1,'left')
        print_tree(node['right'], depth+1,'right')
    else:
        print(depth*' ', node)

# file_name =input("Filename:")
file_name = "project3_dataset1.txt"
# file_name = "project3_dataset2.txt"

data = np.loadtxt(file_name, delimiter="\t",dtype='str')
data = np.array(data)
# print(len(data))

K = 10
maximum_depth = 1000
accuracy = []
precision = []
recall = []
f1 = []

########################
########################
######## DEMO ##########
########################
########################
# tree = get_decision_tree(data,0)
# print_tree(tree,0,'')

ten_fold_cross_valid = np.array_split(data, K)

for i in range(K):
    test_data=ten_fold_cross_valid[i]
    train_data=np.array(np.vstack([x for index,x in enumerate(ten_fold_cross_valid) if index != i]))
    tree = get_decision_tree(train_data,0)

    predicted_labels = get_labels(test_data,tree)
#     print(test_data[:,-1])
#     print(predicted_labels)
    print_tree(tree,0,'')
    TP = FN = FP = TN = 0
    for i in range(len(test_data[:,-1])):
        if test_data[:,-1][i] == '1':
            if predicted_labels[i] == '1':
                TP += 1
            else:
                FN +=1
        else:
            if predicted_labels[i] == '1':
                FP += 1
            else:
                TN +=1
    if TP + FN + FP + TN !=0:
        accuracy.append(float(TP + TN)/(TP + FN + FP + TN))
    if TP + FP !=0:
        precision.append(float(TP)/(TP + FP))
    if TP + FN !=0:
        recall.append(float(TP)/(TP + FN))
    if TP + FN + FP !=0:
        f1.append(float(2 * TP) / ((2 * TP) + FN + FP))

print("Average accuracy  : "+  str(sum(accuracy)*100/len(accuracy)))
print("Average precision : "+  str(sum(precision)*100/len(precision)))
print("Average recall    : "+  str(sum(recall)*100/len(recall)))
print("Average f_measure : "+  str(sum(f1)*100/len(f1)))


# In[ ]:




