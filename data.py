# -*- coding: utf-8 -*-
import sys
import urllib
import urllib2
from bs4 import BeautifulSoup
import re
import string
#import MySQLdb
#import json

url="/Users/lxy/Library/Mobile Documents/com~apple~CloudDocs/ECS/Data Mining/Coursework/Individual/gap-html-2/gap_-C0BAAAAQAAJ/00000015.html"
wp = urllib.urlopen(url)
print "start download..."
content = wp.read()

soul = BeautifulSoup(content)
#identify = string.maketrans('', '')   
#r1 = string.punctuation + ' ' + string.digits
r1 = u'[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
#Find all content in the class "ocr_cinfo" of tag "span"
for i in soul.find_all('span',class_ = "ocr_cinfo"):
	content = i.text
	#content = content.translate(identify,delEStr)

	#content = re.sub(r1, '', content)
	content = re.sub(r1, '', content)
	print content


   #for s in [i, i.parent.find_next_sibling()]:
   	
   	#while s <> None:
      #if s.find('span') <> None:
       # break
      #print 'contents:', s.text
      #s = s.find_next_sibling()
