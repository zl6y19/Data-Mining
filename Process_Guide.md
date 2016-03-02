#文本处理流程
##预处理（对这里的高质量讨论结果的修改，下面的顺序仅限英文）
1. 去掉抓来的数据中不需要的部分，比如 HTML TAG，只保留文本。结合 beautifulsoup 和正则表达式就可以了。pattern.web 也有相关功能。
2. 处理编码问题。没错，即使是英文也需要处理编码问题！由于 Python2 的历史原因，不得不在编程的时候自己处理。英文也存在 unicode 和 utf-8 转换的问题，中文以及其他语言就更不用提了。这里有一个讨论，可以参考，当然网上也有很多方案，找到一个适用于自己的最好。
3. 将文档分割成句子。
4. 将句子分割成词。专业的叫法是 tokenize。
5. 拼写错误纠正。pyenchant 可以帮你！（中文就没有这么些破事！）
6. POS Tagging。nltk 是不二选择，还可以使用 pattern。
7. 去掉标点符号。使用正则表达式就可以。
8. 去掉长度过小的单词。len<3 的是通常选择。
9. 去掉 non-alpha 词。同样，可以用正则表达式完成 \W 就可以。
10. 转换成小写。
11. 去掉停用词。Matthew L. Jockers 提供了一份比机器学习和自然语言处理中常用的停词表更长的停词表。中文的停词表 可以参考这个。
12. lemmatization/stemming。nltk 里面提供了好多种方式，推荐用 wordnet 的方式，这样不会出现把词过分精简，导致词丢掉原型的结果，如果实在不行，也用 snowball 吧，别用 porter，porter 的结果我个人太难接受了，弄出结果之后都根本不知道是啥词了。MBSP 也有相关功能。
13. 重新去掉长度过小的词。是的，再来一遍。重新去停词。上面这两部完全是为了更干净。
14. 到这里拿到的基本上是非常干净的文本了。如果还有进一步需求，还可以根据 POS 的结果继续选择某一种或者几种词性的词。
* Bag-of-Words! nltk 和 scikit.learn 里面都有很完整的方案，自己选择合适的就好。这里如果不喜欢没有次序的 unigram 模型，可以自行选择 bi-gram 和 tri-gram 以及更高的 n-gram 模型。nltk 和 sklearn里面都有相关的处理方法。
* 更高级的特征。
  * TF-IDF。这个 nltk 和 sklearn 里面也都有。
  * Hashing！
* 训练模型
  * 到这里，就根据自己的应用选择合适的学习器就好了。
  * 分类，情感分析等。sklearn 里面很多方法，pattern 里有情感分析的模块，nltk 中也有一些分类器。
  * 主题发现
    * NMF
    * (Online) Latent Dirichlet Allocationword2vec
  * 自动文摘。这个自己写吧，没发现什么成型的工具。
* Draw results
  * Matplotlib
  * Tag cloud
  * Graph
http://zhuanlan.zhihu.com/textmining-experience/19630762