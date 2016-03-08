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



nltk.download("stopwords")
# 遍历指定目录，显示目录下的所有文件名

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
            #Lemmatization
            content_original = content_original + ',' + lemma(i.text)
        # remove all non-alpha characters and convert to lowercase
        content = preprocessWords(content_original)
        # remove stopwords
        content = removestopword(content)
        # remove short words
        content = removeshort(content)

        for word in content:
            out.write('\t%s' % word)
        #out.write('\n')
        print content
    out.close()
    return 1

        # correct wrong spelling problem , act not well
        # try to use TF-IDF to do it
        #content = correctword(content)
        #print content

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

def correctword(words):
    d = enchant.Dict("en_US")
    # check the word by the build-in Dic
    rightwords = ''

    for word in words:
        print d.check(word)
        if d.check(word) == True:
            rightwords = rightwords + ' ' + word
            print word
        else:
            print word
            #suggestion = d.suggest(word)
            #print suggestion
            #rightwords = rightwords + ' ' + suggestion[0]
    return rightwords
    # if true : add to string else check the suggestion

def Tfidf(filelist):

    corpus = []  #存取100份文档的分词结果
    for ff in glob.glob(os.path.join(filelist, "*.txt")):
        f = open(ff,'r+')
        content = f.read()
        f.close()
        corpus.append(content)

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    word = vectorizer.get_feature_names() #所有文本的关键字
    weight = tfidf.toarray()              #对应的tfidf矩阵

    sFilePath = '/Users/lxy/GitHub/Data-Mining/'
    if not os.path.exists(sFilePath) :
        os.mkdir(sFilePath)

    # 这里将每份文档词语的TF-IDF写入tfidffile文件夹中保存
    for i in range(len(weight)) :
        print u"--------Writing all the tf-idf in the",i,u" file into ",sFilePath+'/'+string.zfill(i,5)+'.txt',"--------"
        f = open(sFilePath+'/'+string.zfill(i,5)+'.txt','w+')
        for j in range(len(word)) :
            f.write(word[j]+"    "+str(weight[i][j])+"\n")
        f.close()

    return 1


if __name__ == "__main__" :
    file_path = r"/Users/lxy/Desktop/gap-html/"
    i = 0
    text_path = r"/Users/lxy/GitHub/Data-Mining/"

    #for folds_name in glob.glob(os.path.join(file_path, "*")):
     #   i = i + 1
      #  k = str(i)
       # finalword = getWords(i,folds_name)
        #if finalword == '1':
         #   print "finish write fold" + k

    final_TF = Tfidf(text_path)
    if final_TF == '1':
        print "Success TF-IDF"




