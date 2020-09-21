#!/usr/bin/env python
# coding: utf-8

# In[7]:


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
def normalize_data(data):
    for i in range(len(data[0])-1):
        if (is_Num(data[0][i])):
            X=data[:, i].astype(float)
            data[:, i] = (X - X.min()) / (X.max() - X.min())
    return data
def samples_by_cls(data):
    sample_dict={}
    for sample in data:
        if sample[-1] not in sample_dict:
            sample_dict[sample[-1]]=[]
        sample_dict[sample[-1]].append(sample)
    return sample_dict
def each_attr_summary(data):
    summary=[]
    for nums in zip(*data):
        mean=sum(nums)/float(len(nums))
        stdev=math.sqrt(sum(pow(x-mean,2) for x in nums)/float(len(nums)-1))
        summary.append([mean,stdev])
    del summary[-1]
    return summary
def summary_each_cls(train_data):
    sample_dict=samples_by_cls(train_data)
    summary={}
    for cls,sample in sample_dict.items():
        summary[cls]=each_attr_summary(sample)
    return summary
def cls_Prob(summaries,inputVector,prior_zero,prior_one):
    probs={}
    for cls,summary in summaries.items():
        if(cls==0):
            probs[cls]=prior_zero
        else:
            probs[cls]=prior_one
        for i in range(len(summary)):
            #Gaussian Probability Density Function
            exp=math.exp(-(math.pow(inputVector[i]-summary[i][0],2)/(2*math.pow(summary[i][1],2))))
            probs[cls]*=(1/(math.sqrt(2*math.pi)*summary[i][1]))*exp
    return probs                        
def cal_cat_prob(cat_train_by_cls,cat_test_vector):
    count_zero=[]
    count_one=[]
    prob_one=1
    prob_zero=1
    for cls,sample in cat_train_by_cls.items():
        i=-1
        for nums in zip(*sample):
            i+=1
            if cls=='1':
                count_one.append(nums.count(cat_test_vector[i]))   
            else:
                count_zero.append(nums.count(cat_test_vector[i]))
    for i in range(len(count_one)-1):
        prob_one*=count_one[i]/(len(cat_train_by_cls['1']))
        prob_zero*=count_zero[i]/(len(cat_train_by_cls['0']))
    return prob_one,prob_zero
# file_name =input("Filename:")
# k=int(input("input K:"))
file_name="project3_dataset2.txt"
k=10
data = np.loadtxt(file_name, delimiter="\t",dtype='str')
data =np.asarray(data)
# data=data.astype(np.float)
K_Fold=10
accuracy =[]
precision = []
recall = []
f_measure = []
cat_indices = []
for i in range(len(data[0])):
    if not is_Num(data[0][i]):
        cat_indices.append(i)
cont_data=np.delete(data,cat_indices,1)
cont_data=cont_data.astype(np.float)
ten_fold_cross_valid = np.array_split(cont_data, K_Fold)
cat_indices.append(-1)
data_cat=data[:, cat_indices]
data_cat_ten_fold=np.array_split(data_cat, K_Fold)
for index in range(len(ten_fold_cross_valid)):
    test_data=ten_fold_cross_valid[index]
    train_data=np.array(np.vstack([x for i,x in enumerate(ten_fold_cross_valid) if i != index]))
    summaries=summary_each_cls(train_data)
    result=[]
    prior_zero = float(list(train_data[:,-1]).count(0))/len(train_data)
    prior_one = float(list(train_data[:,-1]).count(1))/len(train_data)
    cat_test=data_cat_ten_fold[index]
    cat_train=np.array(np.vstack([x for i,x in enumerate(data_cat_ten_fold) if i != index]))
    cat_train_by_cls=samples_by_cls(cat_train)
    for i in range(len(test_data)):
        if(len(cat_indices)>1):
            cat_prob_one,cat_prob_zero=cal_cat_prob(cat_train_by_cls,cat_test[i])
        probs=cls_Prob(summaries,test_data[i],prior_zero,prior_one)
        highest_prob=-1
        pred_cls="-1"
        for cls,prob in probs.items():
            if(len(cat_indices)>1):
                if(cls==1):
                    prob*=cat_prob_one
                else:
                    prob*=cat_prob_zero
            if pred_cls=="-1" or prob>highest_prob:
                pred_cls=cls
                highest_prob=prob
        result.append(pred_cls)
    TP = FN = FP = TN = 0
    for i in range(len(test_data[:,-1])):
        if test_data[:,-1][i] == 1:
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





# In[ ]:




