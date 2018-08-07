# encoding:utf8
import gensim

if __name__ == '__main__':

    w2c_model = 'D:\\data\\wiki_embedding\\model\\wiki.zh.model'

    model = gensim.models.Word2Vec.load(w2c_model)
    print('load success!')
    word1 = u'东北'
    word2 = u'瘦脸'
    if word1 in model:
        print (u"'%s'的词向量为： " % word1)
        print (model[word1])
    else:
        print (u'单词不在字典中！')

    result = model.most_similar(word2, topn=20)
    print (u"\n与'%s'最相似的词为： " % word2)
    for e in result:
        print ('%s: %f' % (e[0], e[1]))

    print (u"\n'%s'与'%s'的相似度为： " % (word1, word2))
    print (model.similarity(word1, word2))

    print (u"\n'早餐 晚餐 午餐 中心'中的离群词为： ")
    print (model.doesnt_match(u"早餐 晚餐 午餐 中心".split()))

    print (u"\n与'%s'最相似，而与'%s'最不相似的词为： " % (word1, word2))
    temp = (model.most_similar(positive=[u'篮球'], negative=[u'计算机'], topn=1))
    print ('%s: %s' % (temp[0][0], temp[0][1]))

    list1 = ['面包', '吃', '掉', '3', '招', '保持', '面包', '新鲜', '好味']
    list2 = ['悄悄', '告诉', '你', '英式', '烤鸡', '的', '美味', '秘方']

    list_sim1 = model.n_similarity(list1, list2)
    print(list_sim1)
