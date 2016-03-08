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



nltk.download("stopwords")
# 遍历指定目录，显示目录下的所有文件名

def getWordsTF(file_name):
    input_path = eval(r"file_name")
    for files in glob.glob(os.path.join(input_path, "*.html")):
        fileDir = (files)
        print "file path:" + fileDir
        wp = urllib.urlopen(fileDir)
        print "Start reading file..."
        soul = BeautifulSoup(wp, "html.parser")
        content_original = ''
        for i in soul.find_all('span', class_ = "ocr_cinfo"):
            #Lemmatization
            content_original = content_original + ',' + lemma(i.text)


        # remove all non-alpha characters and convert to lowercase
        content = preprocessWords(content_original)

        # remove stopwords
        content = removestopword(content)

        # remove short words
        content = removeshort(content)
        print content




        #print content
    return content




def preprocessWords(words):


    # Split words by all non-alpha characters
    # 用了这个后出现了u''!!!!!!!!!!!!!!!!!!
    words = re.compile(r'[^A-Z^a-z]+').split(words)

    # Convert to lowercase
    return [word.lower() for word in words if word != '']

def lowerword(words):
    return [word.lower() for word in words if 1 == 1]

def removestopword(words):
    # remove the stop words
    stop = stopwords.words('english')
    return [word for word in words if word not in stop]

def removeshort(words):

    # remove short word
    return [word for word in words if len(word) > 3]

#def correctword(words):
    #d = enchant.Dict("en_GB")
    # check the word by the build-in Dic
    rightwords = ''

    #for word in words:
        #print d.check(word)
        #if d.check(word) == True:
            #rightwords = rightwords + ' ' + word
        #else:
            #suggestion = d.suggest(word)
            #print suggestion
            #rightwords = rightwords + ' ' + suggestion[0]
    #return rightwords
    # if true : add to string else check the suggestion

def Tfidf(filelist) :
    path = './segfile／'
    corpus = []  #存取100份文档的分词结果
    for ff in filelist :
        fname = path + ff
        f = open(fname,'r+')
        content = f.read()
        f.close()
        corpus.append(content)

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    word = vectorizer.get_feature_names() #所有文本的关键字
    weight = tfidf.toarray()              #对应的tfidf矩阵

    sFilePath = './tfidffile'
    if not os.path.exists(sFilePath) :
        os.mkdir(sFilePath)

    # 这里将每份文档词语的TF-IDF写入tfidffile文件夹中保存
    for i in range(len(weight)) :
        print u"--------Writing all the tf-idf in the",i,u" file into ",sFilePath+'/'+string.zfill(i,5)+'.txt',"--------"
        f = open(sFilePath+'/'+string.zfill(i,5)+'.txt','w+')
        for j in range(len(word)) :
            f.write(word[j]+"    "+str(weight[i][j])+"\n")
        f.close()


if __name__ == "__main__" :
    file_path = r"/Users/lxy/Desktop/gap-html/"
    for file_name in glob.glob(os.path.join(file_path, "*")):
        finalword = getWordsTF(file_name)
        print finalword


