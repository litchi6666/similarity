"""
读取文件——结巴分词——保存分词结果
"""
import jieba
import pandas as pd
import re


# corpus = 'D:\\data\\textsimi\\BATresult.csv'
# corpus_after_jieba = 'D:\\data\\textsimi\\jieba_w2c\\tok_result.csv'
# all_title = pd.DataFrame(pd.read_csv(corpus, header=0, encoding='utf-8-sig'))['video_name']

corpus = 'D:\\data\\videos\\videos.csv'
corpus_after_jieba = 'D:\\data\\videos\\tok_result.csv'
all_title = pd.DataFrame(pd.read_csv(corpus, header=0, encoding='utf-8-sig'))['title']

all_title_seg = []

for title in all_title:

    # print(title)

    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~，。、【 】 “”：；（）《》‘’{}？！()、%^>℃：.”“^-——=&#@￥]+'
    title = re.sub(r, '', title)
    seg_result = jieba.cut(title)
    seg_list = ','.join(seg_result)
    all_title_seg.append(seg_list)

title_and_seg = pd.DataFrame({'title': all_title, 'seg': all_title_seg})
title_and_seg.to_csv(corpus_after_jieba, ',', index=False, encoding='utf-8', columns=['title', 'seg'])

print(all_title_seg[3])
print('done!')
