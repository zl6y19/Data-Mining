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

            content_original = content_original + ' ' + lemma(i.text)


        content = preprocessWords(content_original)
        content = removestopword(content)
        content = removeshort(content)

        content = correctword(content)



        # repeat remove stop word and short word



        print content
    return content




def preprocessWords(words):

    # Split words by all non-alpha characters
    words = re.compile(r'[^A-Z^a-z]+').split(words)

    # Convert to lowercase
    return [word.lower() for word in words if word != '']

def removestopword(words):
    # remove the stop words
    stop = stopwords.words('english')
    return [word for word in words if word not in stop]

def removeshort(words):

    # remove short word
    return [word for word in words if len(word) > 3]

def correctword(words):
    d = enchant.Dict("en_GB")
    # check the word by the build-in Dic
    rightwords = ''
    for word in words:
        if d.check(word) == true:
            rightwords = rightwords + ' ' + word
        else:
            rightwords = rightwords + ' ' + d.suggest(word)
    return rightwords
    # if true : add to string else check the suggestion




if __name__ == "__main__" :
    file_path = r"/Users/lxy/Desktop/gap-html/"
    for file_name in glob.glob(os.path.join(file_path, "*")):
        finalword = getWordsTF(file_name)
        print finalword


