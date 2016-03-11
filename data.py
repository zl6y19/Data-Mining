# -*- coding: utf-8 -*-

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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import codecs
from sklearn.cluster import AgglomerativeClustering
import csv


# Read files and call functions
def getWords(i,folds_name):
    input_path = eval(r"folds_name")
    content = ''
    j = str(i)
    files_out = "text" + j
    out = file(files_out + '.txt', 'w')
    for files in glob.glob(os.path.join(input_path, "*.html")):
        fileDir = (files)
        print "file path:" + fileDir
        wp = urllib.urlopen(fileDir)
        print "Start reading file..."
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

        for word in content:
            out.write('\t%s' % word)
        #out.write('\n')
        print content
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
    d_gb = enchant.DictWithPWL("en_GB","/Users/lxy/PycharmProjects/data mining/enwiktionary.txt")
    d_g = enchant.Dict("grc_GR")

    return [word for word in words if d_gb.check(word) or d_g.check(word)]

def Tfidf(filelist):
    # normalization
    # weight
    corpus = []  #get the content from all files
    for ff in glob.glob(os.path.join(filelist, "*.txt")):
        f = open(ff,'r+')
        content = f.read()
        f.close()
        corpus.append(content)

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    # CountVectorizer().fit_transform: Transform the terms into terms frequency matrix
    freq_matrix = vectorizer.fit_transform(corpus)
    tfidf = transformer.fit_transform(freq_matrix)

    word = vectorizer.get_feature_names() #get the key words from all files
    print 'Features length: ' + str(len(word))
    weight = tfidf.toarray()              #对应的tfidf矩阵

    sFilePath = '/Users/lxy/GitHub/Data-Mining/'
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
        #csvfile.close()

    print 'Start Kmeans:'
    clf = KMeans(n_clusters=4)   #景区 动物 人物 国家
    s = clf.fit(weight)
    print s

    #中心点
    print(clf.cluster_centers_)

    #每个样本所属的簇
    label = []               #存储1000个类标 4个类
    print(clf.labels_)
    i = 1
    while i <= len(clf.labels_):
        print i, clf.labels_[i-1]
        label.append(clf.labels_[i-1])
        i = i + 1

    #用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数  958.137281791
    print(clf.inertia_)




    return 1

#def PCA_Graph(matrix):

def k_means(weight):
    print 'Start Kmeans:'
    clf = KMeans(n_clusters=4)   #景区 动物 人物 国家
    s = clf.fit(weight)
    print s

    #中心点
    print(clf.cluster_centers_)

    #每个样本所属的簇
    label = []               #存储1000个类标 4个类
    print(clf.labels_)
    i = 1
    while i <= len(clf.labels_):
        print i, clf.labels_[i-1]
        label.append(clf.labels_[i-1])
        i = i + 1

    #用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数  958.137281791
    print(clf.inertia_)
    return 1


'''def hiercluster(weight):
    pca = PCA(n_components=2)             #输出两维
    newData = pca.fit_transform(weight)   #载入N维
    print newData '''


if __name__ == "__main__" :
    file_path = r"/Users/lxy/Desktop/gap-html/"
    i = 0
    text_path = r"/Users/lxy/GitHub/Data-Mining/"
    #print enchant.list_languages()

    for folds_name in glob.glob(os.path.join(file_path, "*")):
        i = i + 1
        k = str(i)
        finalword = getWords(i,folds_name)
        if finalword == '1':
            print "finish write fold" + k

    weight = Tfidf(text_path)
    #test = k_means(weight)
    print "Finish"





