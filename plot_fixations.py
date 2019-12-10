# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
# from pandas.plotting import scatter_matrix
from pandas.tools.plotting import scatter_matrix
import numpy as np
import csv
from scipy.optimize import curve_fit
from scipy.spatial.distance import euclidean, minkowski, chebyshev, cityblock
import pylab
from scipy import misc
from numpy.random import randn
import math

# 分割数
divide_number = 100

# スライディングウィンドウの窓幅
window_width = 5

norm_threshold = 100

# 注視点の幅を指定
fixation_width = 5

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

# 注視かどうかのラベルづけ
def getLabel(listnorms):
    label = []
    flag = 0
    fixation = 1

    for i in range(0, len(listnorms)):
        # 注視だったら
        if listnorms[i] < norm_threshold:
            label.append(1)
        # 注視じゃなかったら
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

# 注視点ごとにどこの要素番号かを２次元配列で返す
def getFixationsNumber(listxy, label):
    fixation = []
    fixations_list = []
    for i in range(len(listxy)):
        if label[i] == 1:
            fixation.append(i)
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

# 注視の座標の要素番号を一次元配列で返す
def getFixationsNumberOne(number_list):
    numbers = []
    for i in range(0,len(number_list)):
        numbers.extend(number_list[i])
    return numbers

# 生データの座標の平均座標を算出
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
                subList.append(sum(temp)/len(temp))
                flag = False
                temp = []
            subList.append(xylist[i])
    if(temp != []):
        subList.append(sum(temp)/len(temp))
    return subList

def getFixIndex(subfixlist):
    index = []
    for i in range(0,len(subfixlist)):
        if subfixlist[i] < norm_threshold:
            index.append(i)
    return index

# 注視点の平均座標を算出
def getAverageFixations(fixations):
    avgFixations = []
    for i in range(0,len(fixations)):
        avgFixations.append(sum(fixations[i])/len(fixations[i]))
    return avgFixations

# ２フレーム間の変化量を出し，サッケードを抽出
def getSaccades(averageFixations):
    saccades = []
    for i in range(0,len(averageFixations)-1):
        saccades.append(abs(averageFixations[i+1] - averageFixations[i]))
    return saccades

# 縦のサッケードの要素番号を算出
def getLengthSaccadesNumber(x_saccades, y_saccades):
    lengthSaccadesLabel = []
    label = []

    # 縦なら1,横なら-1とラベルづけ
    for i in range(0, len(y_saccades)):
        if y_saccades[i] >= x_saccades[i] :
            label.append(1)
        else :
            label.append(-1)

    for j in range(0,len(label)):
        if label[i] == 1:
            lengthSaccadesLabel.append(j)

    return lengthSaccadesLabel

# 横のサッケードの要素番号を算出
def getWidthSaccadesNumber(x_saccades, y_saccades):
    widthSaccadesLabel = []
    label = []

    # 縦なら1,横なら-1とラベルづけ
    for i in range(0, len(y_saccades)):
        if y_saccades[i] >= x_saccades[i] :
            label.append(1)
        else :
            label.append(-1)

    for j in range(0,len(label)):
        if label[i] == -1:
            widthSaccadesLabel.append(j)

    return widthSaccadesLabel

def main():

    # path = "/Users/motty/Desktop/avg_kiseki.csv"
    path = '/Users/motty/Desktop/for_research/se/5/5matome.csv'
    # path = '/Users/motty/Desktop/fixation_plot.csv'
    save_path = "/Users/motty/Desktop/"
    xlab = "x1"
    ylab = "y1"
    fnam = 'mottys_4.png'

    # csvの内容を全てprint
    # csvfile = open(path)
    # for row in csv.reader(csvfile):
    #     print(row)

    # データの読み込み
    data = pd.read_csv(path)
    # data = getDivideAverage(data)

    # 読み込むデータのラベルを選択
    xlabel = data[xlab]
    ylabel = data[ylab]

    x_label = getDivideAverage(xlabel)
    y_label = getDivideAverage(ylabel)

    norms = slidingWindowCalcNorm(x_label, y_label)
    labels = getLabel(norms)

    fixation_index = getFixationsNumberOne(getFixationsNumber(x_label,labels))

    x_fixation = []
    y_fixation = []

    # x,y座標の中から注視を抽出
    # for i in range(0,len(fixation_index)):
    #     x_fixation.append(x_label[fixation_index[i]])
    #     y_fixation.append(y_label[fixation_index[i]])
    for i in range(0,len(fixation_index)):
        x_fixation.append(x_label[fixation_index[i]])
        y_fixation.append(y_label[fixation_index[i]])
    # print(fixation_index)

    # for i in range(0,len(norms)):
    #     print(norms[i])

    # for i in range(0,len(labels)):
    #     print(labels[i], x_label[i])

    x_fix = getFixations(x_label, labels)
    y_fix = getFixations(y_label, labels)
    x_fix_sub = getFixationSub(x_label, labels)
    y_fix_sub = getFixationSub(y_label, labels)
    print(x_fix_sub)

    x_fix_plt = getAverageFixations(x_fix)
    y_fix_plt = getAverageFixations(y_fix)

    x_saccades = getSaccades(x_fix_plt)
    y_saccades = getSaccades(y_fix_plt)

    x_fix_avg = getAverageFixations(x_fix)
    y_fix_avg = getAverageFixations(y_fix)

    lengthSaccadesNumber = getLengthSaccadesNumber(x_saccades, y_saccades)
    widthSaccadesNumber = getWidthSaccadesNumber(x_saccades, y_saccades)

    y_length_saccades = []
    x_width_saccades = []


    # 背景色の変更
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(1.0)

    # 枠線削除
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # 軸の削除
    plt.axis('off')

    # y軸反転
    plt.gca().invert_yaxis()

    # 縦横比を１：１に
    plt.axes().set_aspect('equal', 'datalim')

    data.describe()

    # 散布図でプロット
    # plt.scatter(x_label, y_label)
    # plt.scatter(x_label, y_label, marker='o', s=50)
    # plt.scatter(x_label, y_label, c='red', s=50)
    # plt.scatter(x_fixation, y_fixation, c = 'blue', s=50)
    # plt.scatter(x_fix_plt, y_fix_plt, c = 'blue', s=100)
    # plt.scatter(x_saccades, y_saccades, c = 'blue', s=100)
    # plt.scatter(x_fix_sub, y_fix_sub, c = 'blue', s=100)

    # 注視の平均座標をプロット
    plt.scatter(x_fix_avg, y_fix_avg, c = 'blue', s=100)

    # for i,(x,y) in enumerate(zip(x_label,y_label), 1):
    #     plt.annotate(str(i),(x,y))
    # for i,(x,y) in enumerate(zip(xplt_data,yplt_data), 1):
    #     plt.annotate(str(i),(x,y))

    r = randn(0).cumsum()
    plt.plot(r, color='y', linestyle='default', marker='o')

    plt.savefig(save_path+fnam, transparent = True, facecolor=fig.get_facecolor(), edgecolor='w' )
    # plt.show()
    # plt.savefig(save_path) # パスを指定してプロット画像を保存

if __name__ == '__main__':
    main()
