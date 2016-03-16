# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import urllib
from bs4 import BeautifulSoup
import re
import string
import os
import glob
from pattern.en import lemma
from nltk.corpus import stopwords
import nltk
import enchant
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import codecs
import csv
import scipy.cluster.hierarchy as hcluster
from sklearn.externals import joblib
import pandas as pd


#nltk.download("stopwords")

# Read files and call functions
def getWords(i,folds_name):
    input_path = eval(r"folds_name")
    content = ''
    j = str(i)
    files_out = "text" + j
    out = file(files_out + '.txt', 'w')
    for files in glob.glob(os.path.join(input_path, "*.html")):
        fileDir = (files)
        print ("file path:" + fileDir)
        wp = urllib.urlopen(fileDir)
        print ("Start reading file...")
        soul = BeautifulSoup(wp, "html.parser")
        content_original = ''
        for i in soul.find_all('span', class_ = "ocr_cinfo"):

            content_original = content_original + ',' + i.text
        # remove all non-alpha characters and convert to lowercase
        content = preprocessWords(content_original)
        # remove short words
        content = removeshort(content)
        # remove misspelling words
        content = correctword(content)
        # remove stopwords
        content = removestopword(content)
        # remove short words
        content = lemmatization(content)
        # remove short words again
        content = removeshort(content)
        # lower words
        content = lowerword(content)

        content = personal_stopword(content)

        for word in content:
            out.write('\t%s' % word)
        #out.write('\n')
        print (content)
    out.close()
    return 1

        # correct wrong spelling problem , act not well
        # try to use TF-IDF to do it

        #print content

def preprocessWords(words):
    # Split words by all non alphabet character
    words = re.compile(r'[^A-Z^a-z]+').split(words)
    # Convert to lowercase
    return [word for word in words if word != '']

def lowerword(words):
    return [word.lower() for word in words if 1 == 1]

def removestopword(words):
    # remove the stop words
    stop = stopwords.words('english')
    return [word for word in words if word not in stop]

def removeshort(words):
    # remove short word
    return [word for word in words if len(word) > 3]

def lemmatization(words):
    #lemmatization
    return [lemma(word) for word in words if word != '']

def correctword(words):
    # call build-in dictionary and self-added dictionary
    pwl = enchant.request_pwl_dict("/Users/lxy/PycharmProjects/data mining/enwiktionary.txt")
    d_gb = enchant.Dict("en_GB")
    d_g = enchant.DictWithPWL("grc_GR", "/Users/lxy/PycharmProjects/data mining/enwiktionary.txt")


    return [word for word in words if d_gb.check(word) or d_g.check(word)]




def Tfidf(filelist):
    # normalization
    # weight
    corpus = []  #get the content from all files
    for ff in glob.glob(os.path.join(filelist, "*.txt")):
        f = open(ff,'r+')
        content = f.read()
        f.close()
        #content = removestopword(content)
        #content = lowerword(content)
        corpus.append(content)

    #vectorizer = CountVectorizer()
    #transformer = TfidfTransformer()
    TfidfVectorizer1 = TfidfVectorizer(max_df = 0.75, min_df = 0.2, stop_words='english')
    # CountVectorizer().fit_transform: Transform the terms into terms frequency matrix
    #freq_matrix = vectorizer.fit_transform(corpus)
    #tfidf = transformer.fit_transform(freq_matrix)

    tfidf = TfidfVectorizer1.fit_transform(corpus)



    #word = vectorizer.get_feature_names() #get the key words from all files
    word = TfidfVectorizer1.get_feature_names() #get the key words from all files
    print ('Features length: ' + str(len(word)))
    weights = tfidf.toarray()              #对应的tfidf矩阵



    '''sFilePath = '/Users/lxy/Desktop/'
    if not os.path.exists(sFilePath):
        os.mkdir(sFilePath)

   # 这里将每份文档词语的TF-IDF写入tfidffile文件夹中保存
    for i in range(len(weight)):
        print u"--------Writing all the tf-idf in the",i,u" file into ",sFilePath+'/'+string.zfill(i,5)+'.txt',"--------"
        f = open(sFilePath+'/'+string.zfill(i,5)+'.txt','w+')
        #csvfile = file('csv_test.csv', 'wb')
        #writer = csv.writer(csvfile)
        #writer.writerow(['text1','text2','text3','text4','text5','text6','text7','text8','text9','text10','text11','text12','text13','text14','text15','text16','text17','text18','text19','text20','text21','text22','text23','text24'])
        for j in range(len(word)):
            f.write(word[j]+"    "+str(weight[i][j])+"\n")
            #writer.writerow(word[j] +','+ str(weight[i][j]))
        f.close()
        #csvfile.close()'''




    return weights, word


def k_means(weights, word):

    #PCA
    #pca = PCA(n_components=200)             #输出两维
    #newData = pca.fit_transform(weights)   #载入N维
    print ('Start Kmeans:')
    true_k = 3
    clf = KMeans(n_clusters=true_k, max_iter=500)   #need to find a good way to set the K
    s = clf.fit(weights)
    #s = clf.fit(newData)
    print (s)


    #中心点
    print(clf.cluster_centers_)

    #每个样本所属的簇
    label = []               #存储1000个类标 4个类
    print(clf.labels_)
    i = 1
    while i <= len(clf.labels_):
        print (i, clf.labels_[i-1])
        label.append(clf.labels_[i-1])
        i = i + 1

    #用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    print(clf.inertia_)



    print("Top terms:")
    order_centroids = clf.cluster_centers_.argsort()[:, ::-1]
    #terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print ("Cluster %d:" % i,)
        for ind in order_centroids[i, :10]:
            print (' %s' % word[ind],)
    print()


    #PCA
    pca = PCA(n_components=3)             #输出两维
    newData = pca.fit_transform(weights)   #载入N维
    print (newData)

    #5A景区
    x1 = []
    y1 = []
    #z1 = []
    i=1
    while i <= len(clf.labels_):
        if clf.labels_[i-1] == 0:
            x1.append(newData[i-1][0])
            y1.append(newData[i-1][1])
            #z1.append(newData[i-1][2])

        i = i + 1

    #动物
    x2 = []
    y2 = []
    #z2 = []
    i=1
    while i <= len(clf.labels_):
        if clf.labels_[i-1] == 1:
            x2.append(newData[i-1][0])
            y2.append(newData[i-1][1])
            #z2.append(newData[i-1][2])

        i = i + 1

    #人物
    x3 = []
    y3 = []
    #z3 = []
    i=1
    while i <= len(clf.labels_):
        if clf.labels_[i-1] == 2:
            x3.append(newData[i-1][0])
            y3.append(newData[i-1][1])
            #z3.append(newData[i-1][2])

        i = i + 1

    #国家
    x4 = []
    y4 = []
    #z4 = []
    i=1
    while i <= len(clf.labels_):
        if clf.labels_[i-1] == 3:
            x4.append(newData[i-1][0])
            y4.append(newData[i-1][1])
            #z4.append(newData[i-1][2])

        i = i + 1

    x5 = []
    y5 = []
    #z5 = []
    i=1
    while i <= len(clf.labels_):
        if clf.labels_[i-1] == 4:
            x5.append(newData[i-1][0])
            y5.append(newData[i-1][1])
            #z5.append(newData[i-1][2])

        i = i + 1

    #四种颜色 红 绿 蓝 黑
    plt.plot(x1, y1, 'or')
    plt.plot(x2, y2, 'og')
    plt.plot(x3, y3, 'ob')
    plt.plot(x4, y4, 'ok')
    plt.plot(x5, y5, 'oy')
    plt.show()

    return 1


'''def hiercluster(weight):
    pca = PCA(n_components=2)             #输出两维
    newData = pca.fit_transform(weight)   #载入N维
    print newData '''


if __name__ == "__main__" :
    file_path = r"/Users/lxy/Desktop/gap-html1/"
    i = 16
    #text_path = r"/Users/lxy/GitHub/Data-Mining/"
    #text_path = r"/Users/lxy/Desktop/results_original/"
    text_path_personal =  r"/Users/lxy/Desktop/test/"
    text_path = r"/Users/lxy/Desktop/test/"


    #print enchant.list_languages()

    '''for folds_name in glob.glob(os.path.join(file_path, "*")):
        i = i + 1
        k = str(i)
        finalword = getWords(i,folds_name)
        if finalword == '1':
            print "finish write fold" + k'''
    #ready = repreprocess(text_path_personal)
    weight, word= Tfidf(text_path)
    test = k_means(weight, word)
    print ("Finish")




