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
from sklearn.svm import OneClassSVM


mmsc = MinMaxScaler()

stdsc = StandardScaler()

# 読み込むcsv
m_csv_path = '/Users/motty/Desktop/for_github/python_files/read_data/motty_summary_120.csv'
a_csv_path = '/Users/motty/Desktop/for_github/python_files/read_data/aochi_summary.csv'
r_csv_path = '/Users/motty/Desktop/for_github/python_files/read_data/ryonryon_summary.csv'
i_csv_path = '/Users/motty/Desktop/for_github/python_files/read_data/iwaken_summary.csv'
w_csv_path = '/Users/motty/Desktop/for_github/python_files/read_data/watanabe_summary.csv'

num = 1

# 分割数
divide_number = 100

# 特徴数
# fn = 20 + ((divide_number-1)*2)
fn = 27
# fn =2
# fn =(divide_number-1)*2


# スライディングウィンドウの窓幅
window_width = 3

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

# 平均座標からスライディングウィンドウしてノルムを格納した配列を返す
def slidingWindowCalcNorm(listx, listy):
    norms = []
    for i in range(len(listx)-window_width):
        sumx = pow((listx[i+1]-listx[i]), 2)
        sumy = pow((listy[i+1]-listy[i]), 2)
        norms.append(math.sqrt(sumx + sumy))
    for j in range(0, window_width):
        norms.append(norms[-1])
    norms[0] = norms[1]

    return norms

# 注視華道家のラベルづけ
def getLabel(listnorms):
    label = []
    flag = 0
    fixation = 1

    for i in range(0, len(listnorms)):
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

# 平均座標から注視のサブリストとサッケードの要素を入れた配列を返す
def getFixationSub(xylist, fixlabels):
    subList = []
    temp = []
    flag = False
    for i in range(len(fixlabels)-1):
        if(fixlabels[i] == 1):
            temp.append(xylist[i])
            flag = True
        if(fixlabels[i] == -1):
            if(flag):
                subList.append(temp)
                flag = False
                temp = []
            subList.append(xylist[i])
    if(temp != []):
        subList.append(temp)
    return subList

# サッケードの速度を算出
def getSaccadeSpeed(subList):
    speeds = []
    saccade_length = []
    fixs = []
    flag = False
    count = 0

    for i in range(len(subList)):
        if(type(subList[i]) is list):
            if (count != 0):
                saccade_length.append(count)
                count = 0
            flag = True
            fixs.append(sum(subList[i])/len(subList[i]))
        else:
            if(flag):
                flag = False
                count = count+1
    for j in range(len(saccade_length)):
        speeds.extend(abs((fixs[j+1] - [j])/saccade_length[j]))
    return speeds


# 注視点の平均座標を算出
def getAverageFixations(fixations):
    avgFixations = []
    for i in range(0,len(fixations)):
        avgFixations.append(sum(fixations[i])/len(fixations[i]))
    return avgFixations

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
    reader = int(len(filename.columns)/2)

    for row in range(1,reader+1):
        # 読み込むカラム
        xname = 'x'+str(row)
        yname = 'y'+str(row)
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


        # 注視点の平均座標を算出
        x_fix_avg = getAverageFixations(xfixs)
        y_fix_avg = getAverageFixations(yfixs)

        x_fix_sub = getFixationSub(x, fixation_labels)
        y_fix_sub = getFixationSub(y, fixation_labels)

        # サッケードの速度の算出
        xspeeds = getSaccadeSpeed(x_fix_sub)
        yspeeds = getSaccadeSpeed(y_fix_sub)

        xspeeds_avg = sum(xspeeds)/len(xspeeds)
        yspeeds_avg = sum(yspeeds)/len(yspeeds)

        xspeeds_max = max(xspeeds)
        yspeeds_max = max(yspeeds)

        xspeeds_min = min(xspeeds)
        yspeeds_min = min(yspeeds)

        saccade_count = len(xspeeds)


        # print(fixation_labels)
        # print(xspeeds)

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
        feature.append(xspeeds_avg)
        feature.append(yspeeds_avg)
        feature.append(xspeeds_max)
        feature.append(yspeeds_max)
        feature.append(xspeeds_min)
        feature.append(yspeeds_min)
        feature.append(saccade_count)

    # 配列に叩き込むルート
    result = [feature[i:i+fn] for i in range(0, len(feature), fn)]

    return np.array(result, np.float32)

def ocsvm_gridSearch(train, owner, attacker, nu_list, gamma_list):
    opt_nu = 0
    opt_gamma = 0
    max_accuracy = 0

    # グリッドサーチ
    for nu in nu_list:
        for gamma in gamma_list:
            clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
            clf.fit(train)
            p_o = clf.predict(owner)
            p_a = clf.predict(attacker)
            tp = p_o[p_o == 1].size # 正しく本人を受け入れた
            tn = p_a[p_a == -1].size # 正しく他人を拒否した
            fp = p_a[p_a == 1].size # 誤って他人を受け入れた
            fn = p_o[p_o == -1].size # 誤って本人を拒否した
            accuracy = (tp + tn)/(tp + tn + fp + fn)
            if max_accuracy < accuracy:
                max_accuracy = accuracy
                opt_nu = nu
                opt_gamma = gamma

    print("best nu is ", opt_nu)
    print("best gamma is ", opt_gamma)
    # 適切なパラメータでの分類器作成
    clf = OneClassSVM(nu=opt_nu, kernel='rbf', gamma=opt_gamma)
    clf.fit(owner)

    # 本人かを識別
    p_o = clf.predict(owner)
    # 他人かを識別
    p_a = clf.predict(attacker)

    # print('owner score', p_o)
    # print('attacker score', p_a)

    tp = p_o[p_o == 1].size # 正しく本人を受け入れた
    tn = p_a[p_a == -1].size # 正しく他人を拒否した
    fp = p_a[p_a == 1].size # 誤って他人を受け入れた
    fn = p_o[p_o == -1].size # 誤って本人を拒否した
    print(tp, tn, fp, fn)
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = (2*precision*recall) / (precision + recall)
    print("precision:", precision)
    print("recall:", recall)
    print("f-measure:", fmeasure)
    print("accuracy:", accuracy)
    print('\n')

    return [precision, recall, accuracy, fmeasure]


def main():

    mf = getFeatures(m_csv_path, 'm')
    af = getFeatures(a_csv_path, 'a')
    rf = getFeatures(r_csv_path, 'r')
    iwf = getFeatures(i_csv_path, 'i')
    wf = getFeatures(w_csv_path, 'w')

    features = np.concatenate([mf, af, rf, iwf, wf])
    owner = mf
    attacker = np.concatenate([af, rf, iwf, wf])

    x_f = features
    x = owner

    # OneClassSVMのハイパーパラメータ
    nu_list = [0.0001, 0.000001, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.7]
    gamma_list = [0.0001, 0.0002, 0.0003, 0.001, 0.01]

    ocsvm_gridSearch(owner, owner, attacker, nu_list, gamma_list)

if __name__ == '__main__':
    # i = 1
    # for i in range(1, 100):
    #     divide_number = i
    main()
