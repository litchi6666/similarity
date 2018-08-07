"""
结合分词，词向量，生成句子向量
将句子向量生成CSV文件保存
考虑是否有词向量的权重构建，weight使用tfidf
"""
from simi import word_embedding
import numpy as np
import pandas as pd


csv_file = 'D:\\data\\textsimi\\BATresult.csv'  # 已经分好词的分词文件
title_vec_file = 'D:\\data\\textsimi\\matrix_title_vec.csv'  # 要保存的标题向量文件


def get_title_vec(sentence_with_seg):
    """
    标题向量的计算
    :param sentence_with_seg:分好词的一句话
    :return:
    """
    try:
        title_vec = np.zeros(1024)  # 标题的向量

        if sentence_with_seg is np.nan:  # 有些句子未生成词向量，分词结果为空，pandas中为nan
            return title_vec

        words_list = sentence_with_seg.split(',')  # 获取句子的每一个分词
        for word in words_list:  # 对每一个分词获取词向量
            if word in word_embed.keys():
                word_vec = word_embed[word]  # 获取词向量
                title_vec += word_vec  # 标题向量等于每一个词向量相加
            # else:
            #     dict_no_vec_word[word] = 1
            #     print(word)  # 没有词向量的词语展示
        return title_vec
    except BaseException as e:
        print(sentence_with_seg)
        print(e)


if __name__ == '__main__':
    dict_no_vec_word = {}
    try:
        word_embed = word_embedding.get_embeding_from_mysql()  # 加载词向量

        df = pd.DataFrame(pd.read_csv(csv_file, header=0))  # 加载分词结果

        baidu_seg_list = []
        tencent_seg_list = []
        ali_seg_list = []

        for i in range(len(df['video_name'])):  # 循环每一行，根据词向量生成句子向量
            baidu_seg_vec = get_title_vec(df['baidufc'][i])  # 由百度分词生成的词向量
            tencent_seg_vec = get_title_vec(df['tencentfc'][i])
            ali_seg_vec = get_title_vec(df['alifc'][i])

            baidu_seg_list.append(baidu_seg_vec.tolist())
            tencent_seg_list.append(tencent_seg_vec.tolist())
            ali_seg_list.append(ali_seg_vec.tolist())

            if i % 100 == 0:
                print(i)

        dataframe = pd.DataFrame({'video_name': df['video_name'],
                                  'baidu_seg_vec': baidu_seg_list,
                                  'tencent_seg_vec': tencent_seg_list,
                                  'ali_seg_vec': ali_seg_list})
        col = ['video_name', 'baidu_seg_vec', 'tencent_seg_vec', 'ali_seg_vec']
        dataframe.to_csv(title_vec_file, index=False, sep=',', columns=col)

        print('done!')
        #
        # with open('d:\\a.txt', 'w', encoding='utf-8') as ff:
        #     keys = dict_no_vec_word.keys()
        #     ff.write(str(len(keys)))
        #     ff.write('\n')
        #     for word in keys:
        #         ff.write(word)
        #         ff.write('\n')
        #


    except:
        print(i)
        print(df['video_name'][i])
        print(df['baidufc'][i])  # 由百度分词生成的词向量
        print(df['tencentfc'][i])
        print(df['alifc'][i])

