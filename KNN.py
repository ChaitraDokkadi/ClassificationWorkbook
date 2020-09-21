#!/usr/bin/env python
# coding: utf-8

# In[26]:


import random
import math 
import sys
import os
import numpy as np

def is_Num(data):
    try:
        float(data)
        return True
    except:
        return False
# def normalize_data(data):
#     for i in range(len(data[0])-1):
#         if (is_Num(data[0][i])):
#             X=data[:, i].astype(float)
#             data[:, i] = (X - X.min()) / (X.max() - X.min())
#     return data
def eucledian_distance(train,test):
    dist = 0
    for i in range(len(test)):
        if (is_Num(train[i])):
            dist += ((train[i].astype(float) - test[i].astype(float)) ** 2)
        else:
            dist += 1.0 if test[i] != train[i] else 0.0
    return math.sqrt(dist)
def knn(train_data,test_data):
    result=[]
    for i in range(len(test_data)):
        distance_list = []
        for train_ind in range(len(train_data)):
            dist=eucledian_distance(train_data[train_ind][0:-1],test_data[i][0:-1])
            distance_list.append(dist)
        nearest_points = np.argsort(np.asarray(distance_list))[:k]
        class_one=0
        for point in nearest_points:
            if train_data[point][-1]=='1':
                class_one+=1
        if(class_one>(len(nearest_points)-class_one)):
            result.append(1)
        else:
            result.append(0)
    return result
# file_name =input("Filename:")
k=int(input("input K:"))
file_name="project3_dataset2.txt"
# k=9
data = np.loadtxt(file_name, delimiter="\t",dtype='str')
data =np.asarray(data)
ground_truth = data[:, -1]
K_Fold=10
ten_fold_cross_valid = np.array_split(data, K_Fold)
accuracy =[]
precision = []
recall = []
f_measure = []
for index in range(len(ten_fold_cross_valid)):
    test_data=ten_fold_cross_valid[index]
    train_data=np.array(np.vstack([x for i,x in enumerate(ten_fold_cross_valid) if i != index]))
#     train_data=normalize_data(train_data)
    result=knn(train_data,test_data)
    TP = FN = FP = TN = 0
    for i in range(len(test_data[:,-1])):
        if test_data[:,-1][i] == '1':
            if result[i] == 1:
                TP += 1
            else:
                FN +=1
        else:
            if result[i] == 1:
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
        f_measure.append(float(2 * TP) / ((2 * TP) + FN + FP))
print("Average accuracy  : "+  str(sum(accuracy)*100/len(accuracy)))
print("Average precision : "+  str(sum(precision)*100/len(precision)))
print("Average recall    : "+  str(sum(recall)*100/len(recall)))
print("Average f_measure : "+  str(sum(f_measure)*100/len(f_measure)))


# In[ ]:




