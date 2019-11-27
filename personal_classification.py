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


mmsc = MinMaxScaler()

stdsc = StandardScaler()

# 読み込むcsv
m_csv_path = '/Users/motty/Desktop/for_research/se/5/5matome.csv'
a_csv_path = '/Users/motty/Desktop/for_research/measure_data/0319/aochi/aochi_read_sum.csv'
r_csv_path = '/Users/motty/Desktop/for_research/measure_data/0319/ryonryon/read_sum.csv'
i_csv_path = '/Users/motty/Desktop/for_research/measure_data/2019_0611/iwaken/iwaken_read_sum.csv'
w_csv_path = '/Users/motty/Desktop/for_research/measure_data/2019_0611/watanabe/watanabe_read_sum.csv'

num = 1

# 分割数
divide_number = 100

# 特徴数
# fn = 20 + ((divide_number-1)*2)
fn = 20
# fn =2
# fn =(divide_number-1)*2


# スライディングウィンドウの窓幅
window_width = 5

# 注視点だと判別するための閾値
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
    for j in range(0, window_width):
        norms.append(norms[-1])

    norms[0] = norms[1]

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



    return label

# x,y座標の注視点を二次元配列で取得
def getFixations(listxy, label):
    fixation = []
    fixations_list = []
    for i in range(len(listxy)):
        if label[i] == 1:
            fixation.append(listxy[i])
        else:
            # if(fixation != []):
                # 注視点候補箇所の幅が閾値以上だったら
            if len(fixation) >= fixation_width:
                fixations_list.append(fixation)
            fixation = []
    if(fixation != []):
        fixations_list.append(fixation)
        fixation = []

    return fixations_list

# 二次元配列から分散を算出
def getListVar(list):
    listvar = []
    listcalc = []
    for i in range(len(list)):
        listcalc.append(np.var(list[i]))
    if listcalc == []:
        listcalc = [0,0,0]

    # 分散の最大値・最小値・平均値算出
    if listcalc != []:
        listvar.append(max(listcalc))
        listvar.append(min(listcalc))
        listvar.append(sum(listcalc) / len(listcalc))
    else:
        listvar = [0,0,0]

    return listvar

# 二次元配列から標準偏差を算出
def getListStd(list):
    liststd = []
    listcalc = []
    for i in range(len(list)):
        listcalc.append(np.std(list[i]))

    # 標準偏差の最大値・最小値・平均値算出
    if listcalc == []:
        liststd = [0,0,0]
    else:
        liststd.append(max(listcalc))
        liststd.append(min(listcalc))
        liststd.append(sum(listcalc) / len(listcalc))

    return liststd

# 複数の注視から注視の平均時間，最大時間を算出
def getFixationTime(list):
    times = []
    fixation_time = []
    for i in range(len(list)):
        times.append(len(list[i]))
    if fixation_time != []:
        fixation_time.append(sum(times) / len(times))
        fixation_time.append(max(times))
    else:
        fixation_time = [0,0]

    return fixation_time

def getFeatures(csv_path, name):

    filename = pd.read_csv(csv_path)
    feature = []

    for num in range(1,31):

        # 読み込むカラム
        xname = 'x'+str(num)
        yname = 'y'+str(num)
        listx = []
        listy = []
        listx = filename[xname]
        listy = filename[yname]

        # NaN(Null)削除
        listx = listx[np.logical_not(np.isnan(listx))]
        listy = listy[np.logical_not(np.isnan(listy))]

        x = getDivideAverage(listx)
        y = getDivideAverage(listy)

        # 分散をbunに格納
        bunx = np.var(listx)
        buny = np.var(listy)

        # 標準偏差
        hyox = np.std(listx)
        hyoy = np.std(listy)

        xac = getAmountChange(x)
        yac = getAmountChange(y)

        draw_time = len(listx)

        # スライディングウィンドウでノルム算出
        norms = slidingWindowCalcNorm(x, y)
        # ノルムから注視点のラベルを算出
        fixation_labels = getLabel(norms)
        # x, y座標の注視を抽出
        xfixs = getFixations(x, fixation_labels)
        yfixs = getFixations(y, fixation_labels)
        # print(fixation_labels)
        # print(xfixs)

        # 注視点の分散を算出
        xfixs_var = getListVar(xfixs)
        yfixs_var = getListVar(yfixs)

        #注視点の標準偏差を算出
        xfixs_std = getListStd(xfixs)
        yfixs_std = getListStd(yfixs)

        # 注視時間の特徴量を抽出
        fixation_time = getFixationTime(xfixs)

        # 注視回数を抽出
        fixation_count = len(xfixs)

        # ここで抽出した特徴量をぶち込む
        # feature.extend(xac)
        # feature.extend(yac)
        feature.append(bunx)
        feature.append(buny)
        feature.append(hyox)
        feature.append(hyoy)
        feature.append(draw_time)
        feature.extend(fixation_time)
        feature.extend(xfixs_var)
        feature.extend(yfixs_var)
        feature.extend(xfixs_std)
        feature.extend(yfixs_std)
        feature.append(fixation_count)

    # 配列に叩き込むルート
    result = [feature[i:i+fn] for i in range(0, len(feature), fn)]

    return np.array(result, np.float32)

# confusion matrixからtp,tn,fp,fnの数を算出
# def calc_posinega(matrix):
#     i = 0
#     j = 0
#     tptnfpfn = [0, 0, 0, 0]
#     for i in range(0,len(matrix)-1):
#         for j in range(0,len(matrix)-1):
#             if(i>j)

def main():

    mf = getFeatures(m_csv_path, 'm')
    af = getFeatures(a_csv_path, 'a')
    rf = getFeatures(r_csv_path, 'r')
    iwf = getFeatures(i_csv_path, 'i')
    wf = getFeatures(w_csv_path, 'w')

    features = np.concatenate([mf, af, rf, iwf, wf])

    # test_features = features

    labels = np.array([
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4
    ])

    # labels = np.array([
    # 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    # 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    # 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    # ])

    # labels = np.array([
    # 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    # 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    # 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    # 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    # 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
    # ])

    x = features
    y = labels

    total_accuracy = total_importances = total_confusion_matrix = total_f_measure = 0
    num_splits = 0

    skf = StratifiedKFold(n_splits=3)
    total_predicted = []

    tp = 0
    tn = 0
    fp = 0
    fn = 0


    for train_index, test_index in skf.split(x, y):
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]

        # 訓練データを標準化
        # x_train = stdsc.fit_transform(x_train)
        # x_test = stdsc.fit_transform(x_test)
        # x_train = stdsc.transform(x_train)
        # x_test = stdsc.transform(x_test)

        # 訓練データを正規化
        # x_train = mmsc.fit_transform(x_train)
        # x_test = mmsc.fit_transform(x_test)
        # x_train = mmsc.transform(x_train)
        # x_test = mmsc.transform(x_test)


        # SVM
        # clf = svm.SVC(C=10, gamma=0.1)
        # clf.fit(x_train, y_train)
        # expected = y_test
        # predicted = clf.predict(x_test)

        # RandomForest
        clf = ensemble.RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=1)
        clf.fit(x_train, y_train)
        expected = y_test
        predicted = clf.predict(x_test)

        #　一つ一つのデータがどのクラスに分類されたかの中身
        # print(predicted)


        total_confusion_matrix += confusion_matrix(expected, predicted)
        total_importances += clf.feature_importances_

        total_accuracy += accuracy_score(expected, predicted)
        f_measure = precision_recall_fscore_support(expected, predicted, average='macro')
        # tp+= confusion_matrix(expected, predicted).ravel()
        # fp+= confusion_matrix(expected, predicted).ravel()
        # fn+= confusion_matrix(expected, predicted).ravel()
        # tn+= confusion_matrix(expected, predicted).ravel()

        # tp, fp, fn, tn = confusion_matrix(expected, predicted).ravel()
        # print(f_measure)
        total_f_measure += f_measure[2]
        num_splits += 1
        # EER = bob.measure.eer_threshold()

        # print(total_f_measure)

        # print('Accuracy:\n', accuracy_score(expected, predicted))

        # print('F-measure:\n', f1_score(expected, predicted, average='micro'))
    # confusion matrix を一行にならす
    evaluation = total_confusion_matrix.ravel()
    # print(total_confusion_matrix)

    # tp = evaluation[0]
    # fn = evaluation[1]
    # fp = evaluation[2]
    # tn = evaluation[3]
    #
    # far = fp / (tn + fp)
    # frr = fn / (fn + tp)
    # print(far, frr)

    # tp, fn, fp, tn = .ravel()
    # print(tp, fp, fn, tn)
    # print(tp)
    # print('Divide number:', divide_number)
    print('Total accuracy:', total_accuracy / num_splits)
    print('F-measure:', round(total_f_measure / num_splits, 3))
    print('Total confusion matrix:\n', total_confusion_matrix)


    # 特徴量全ての変数重要度
    # for i in range(0, divide_number-1):
    #     print('x[' + str(i) + ']:', total_importances[i])
    #
    # ind = 0
    # for j in range(divide_number-1, divide_number*2-2):
    #     print('y[' + str(ind) + ']:', total_importances[j])
    #     ind = ind+1
    # print('xvar:', total_importances[divide_number*2-2])
    # print('yvar:', total_importances[divide_number*2-1])
    # print('stdev:', total_importances[divide_number*2])
    # print('ystdev:', total_importances[divide_number*2+1])
    # print('draw_time:', total_importances[divide_number*2+2])
    # print('fixation_time_max:', total_importances[divide_number*2+3])
    # print('fixation_time_average:', total_importances[divide_number*2+4])
    # print('fixation_xvar_max:', total_importances[divide_number*2+5])
    # print('fixation_xvar_min:', total_importances[divide_number*2+6])
    # print('fixation_xvar_average:', total_importances[divide_number*2+7])
    # print('fixation_yvar_max:', total_importances[divide_number*2+8])
    # print('fixation_yvar_min:', total_importances[divide_number*2+9])
    # print('fixation_yvar_average:', total_importances[divide_number*2+10])
    # print('fixation_xstv_max:', total_importances[divide_number*2+11])
    # print('ixation_xstv_min:', total_importances[divide_number*2+12])
    # print('fixation_xstv_average:', total_importances[divide_number*2+13])
    # print('fixation_ystv_max:', total_importances[divide_number*2+14])
    # print('fixation_ystv_min:', total_importances[divide_number*2+15])
    # print('fixation_ystv_average:', total_importances[divide_number*2+16])
    # print('fixation_count:', total_importances[divide_number*2+17])

    print('xvar:', total_importances[0])
    print('yvar:', total_importances[1])
    print('stdev:', total_importances[2])
    print('ystdev:', total_importances[3])
    print('draw_time:', total_importances[4])
    print('fixation_time_max:', total_importances[5])
    print('fixation_time_average:', total_importances[6])
    print('fixation_xvar_max:', total_importances[7])
    print('fixation_xvar_min:', total_importances[8])
    print('fixation_xvar_average:', total_importances[9])
    print('fixation_yvar_max:', total_importances[10])
    print('fixation_yvar_min:', total_importances[11])
    print('fixation_yvar_average:', total_importances[12])
    print('fixation_xstv_max:', total_importances[13])
    print('ixation_xstv_min:', total_importances[14])
    print('fixation_xstv_average:', total_importances[15])
    print('fixation_ystv_max:', total_importances[16])
    print('fixation_ystv_min:', total_importances[17])
    print('fixation_ystv_average:', total_importances[18])
    print('fixation_count:', total_importances[19])

    print('\n')

if __name__ == '__main__':
    # i = 1
    # for i in range(1, 100):
    #     divide_number = i
    main()
