# -*- coding: utf-8 -*
import os, glob, sys
import pandas as pd
import numpy as np
import csv
import math
import os
from sklearn import svm, ensemble
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.model_selection import KFold

# 読み込むcsv
m_csv_path = '/Users/motty/Desktop/for_research/se/5/5matome.csv'

# 分割数
divide_number = 100

# スライディングウィンドウの窓幅
window_width = 3

norm_threshold = 100

# 注視点の幅を指定
fixation_width = 5

# 分割・平均化した座標を算出
def getDivideAverage(list):
    index = 0
    avg = []
    # ほぼ等しく分割
    calc_number_pre = [int((len(list) + i) / divide_number) for i in range(divide_number)]
    for i in range(len(calc_number_pre)):
        sub_len = calc_number_pre[i]
        sum = 0
        for j in range(index, index + sub_len):
            sum += list[j]
        avg.append(round((sum/sub_len), 2))
        # print(avg[i])
        index += calc_number_pre[i]
    return avg

# 変化量を計算
def getAmountChange(list):
    ac = []
    for k in range(0, len(list)-1):
        ac.append(list[k+1]-list[k])
    return ac

def slidingWindowCalcNorm(listx, listy):
    norms = []

    for i in range(len(listx)-window_width+1):
        sumx = 0
        sumy = 0
        subnorms = []
        for j in range(i, i + window_width):
            sumx += (listx[j]-listx[j-1])**2
            sumy += (listy[j]-listy[j-1])**2
        norms.append(math.sqrt(sumx + sumy))

    return norms

def getLabel(listnorms):
    label = []
    flag = 0
    fixation = 1

    for i in range(0, len(listnorms)-1):
        if listnorms[i] < norm_threshold:
            label.append(fixation)
        else:
            label.append(-1)

    #分割数に合わせるために窓幅分だけラベルを格納した配列の拡張(末尾の値を入れる)
    for i in range(window_width):
        label.append(label[-1])

    return label

def getFixations(listxy, label):
    fixation = []
    fixations_list = []
    for i in range(len(listxy)):
        if label[i] == 1:
            fixation.append(listxy[i])
        else:
            if(fixation != []):
                fixations_list.append(fixation)
                fixation = []

    return fixations_list


def getNorms(csv_path, name):

    filename = pd.read_csv(csv_path)
    feature = []

    # for num in range(1,31):

    # 読み込むカラム
    xname = 'x'+str(1)
    yname = 'y'+str(1)
    listx = []
    listy = []
    listx = filename[xname]
    listy = filename[yname]

    # NaN(Null)削除
    listx = listx[np.logical_not(np.isnan(listx))]
    listy = listy[np.logical_not(np.isnan(listy))]

    # 分割した座標取得
    x = getDivideAverage(listx)
    y = getDivideAverage(listy)

    # 生の座標を取得
    # x = listx
    # y = listy

    # 座標算出
    # for i in range(0,len(x)):
    #     print(x[i])
    # for i in range(0,len(y)):
    #     print(y[i])

    result = slidingWindowCalcNorm(x, y)
    motty_labels = getLabel(result)
    print(motty_labels)
    fixs = getFixations(x, motty_labels)
    # return np.array(result, np.float32)
    return fixs

def main():

    motty_norms = getNorms(m_csv_path, 'm')
    print(motty_norms)


    # 計算したノルムを表示
    for i in range(len(motty_norms)):
        print(motty_norms[i])
    # print(len(motty_labels))


if __name__ == '__main__':
    main()
