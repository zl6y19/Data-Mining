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
from sklearn.manifold import MDS
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import codecs
import csv
import scipy.cluster.hierarchy as hcluster
from sklearn.externals import joblib
import pandas as pd
from scipy.cluster.hierarchy import ward, dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import seaborn as sns
from sklearn import manifold
from matplotlib.collections import LineCollection

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

    # Set the parameters of TF-IDF
    TfidfVectorizer1 = TfidfVectorizer(max_df = 0.65, min_df = 0.4, stop_words='english')

    # Transform the terms into terms frequency matrix
    tfidf = TfidfVectorizer1.fit_transform(corpus)


    word = TfidfVectorizer1.get_feature_names() #get the key words from all files
    print ('Features length: ' + str(len(word)))
    weights = tfidf.toarray()              # get TF-IDF matrix


    # produce the warm-cold diagram for the similarity between texts
    '''norm = np.linalg.norm(weights,axis=1).reshape((weights.shape[0],1))
    weights2 = np.dot(weights, weights.T) / np.dot(norm, norm.T)
    sns.heatmap(weights2, annot=True, center=0, cmap='coolwarm') # RdBu_r coolwarm
    plt.title('Similarity by TF-IDF', fontsize=60)
    plt.savefig('similarity.png', dpi=500, figsize=(2000, 2000))
    plt.show()
    plt.close()'''




    return weights, word


def k_means(weights, word):


    print ('Start Kmeans:')
    true_k = 5
    # Set the parameter of K-means
    clf = KMeans(n_clusters=true_k, max_iter=500, n_init=50)   #need to find a good way to set the K
    s = clf.fit(weights)
    print (s)


    # print centroid points
    print(clf.cluster_centers_)

    # Print clusters for each samples
    label = []
    print(clf.labels_)
    i = 1
    while i <= len(clf.labels_):
        print (i, clf.labels_[i-1])
        label.append(clf.labels_[i-1])
        i = i + 1

    # evaluate the number of clusters
    print(clf.inertia_)


    # Print top terms
    print("Top terms:")
    order_centroids = clf.cluster_centers_.argsort()[:, ::-1]
    for i in range(true_k):
        print ("Cluster %d:" % i,)
        for ind in order_centroids[i, :10]:
            print (' %s' % word[ind],)
            print (weight[i][ind])
    print()


    #PCA
    #pca = PCA(n_components=3)             # Set the output dimension
    #newData = pca.fit_transform(weights)   # Put the data in
    #print (newData)
    #MDS
    mds = manifold.MDS(n_components=2, dissimilarity='euclidean')
    newData = mds.fit_transform(weights)

    #visualisation
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

    # produce the diagram
    plt.title('K-means Clustering with PCA', fontsize=20)
    plt.xlabel('Dimension 1', fontsize=15)
    plt.ylabel('Dimension 2', fontsize=15)

    plt.plot(x1, y1, 'or')
    plt.plot(x2, y2, 'og')
    plt.plot(x3, y3, 'ob')
    plt.plot(x4, y4, 'ok')
    plt.plot(x5, y5, 'oy')
    #plt.savefig('k-means.png', dpi=500)
    plt.show()
    plt.close()

    return 1


def hiercluster(weight, word):
    linkage_matrix = ward(weight) #define the linkage_matrix using ward clustering pre-computed distances
    fig, ax = plt.subplots(figsize=(15, 10)) # set size
    ax = dendrogram(linkage_matrix, orientation="top", leaf_font_size=15.);



    plt.title('Hierarchical Clustering' ,fontsize=30)
    plt.xlabel('Text index ', fontsize=20)
    plt.ylabel('Distance', fontsize=20)
    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='on',      # ticks along the bottom edge are on
        top='on',         # ticks along the top edge are on
        labelbottom='on')

    plt.tight_layout() #show plot with tight layout


    plt.savefig('ward_clusters.png', dpi=400) #save figure as ward_clusters
    plt.show()
    plt.close()

# Function for testing the MDS results
def MDS(weight):
    clf = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')
    pos = clf.fit_transform(weight)
    #clf = PCA(n_components=2)
    #mdsf = clf.fit_transform(mdsf)


    # produce the diagram
    plt.title('MDS', fontsize=20)
    plt.xlabel('Dimension 1', fontsize=15)
    plt.ylabel('Dimension 2', fontsize=15)

    plt.scatter(pos[:, 0], pos[:, 1], s=20, c='g')
    plt.savefig('mds.png', dpi=500)
    plt.show()
    plt.close()



    return 1



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
    #test3 = MDS(weight)
    test = k_means(weight, word)
    test2 = hiercluster(weight, word)
    print ("Finish")





