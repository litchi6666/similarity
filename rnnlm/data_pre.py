import pandas as pd
import re
import jieba

file = pd.read_csv('D:/data/textsimi/BATresult.csv')
title = file['video_name']

title_list = []
with open('D:/data/videos/videos_test.txt', 'w', encoding='utf-8-sig') as ff:
    for line in title:
        r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~，。、【 】 “”：；（）《》‘’{}？！()、%^>℃：.”“^-——=&#@￥|' \
            '1234567890]+'
        line = re.sub(r, '', line)
        ff.write(' '.join(jieba.cut(line)))
        ff.write('\n')

print('done')
