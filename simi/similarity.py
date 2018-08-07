from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


#vec_file = 'D:\\data\\textsimi\\matrix_title_vec.csv'

vec_file = 'D:\\data\\textsimi\\BATresult_vec.csv'


def load_vec(which_seg):
    """
    加载标题+词向量
    :param which_seg: 使用的词向量
    :return: 标题，词向量
    """
    seg_list=['baidu_seg_vec', 'tencent_seg_vec', 'ali_seg_vec','title_vec']  # video_name
    if which_seg not in seg_list:
        print('no such seg')
        return
    print('#' * 10 + ' load data..' + '#' * 10)
    df = pd.DataFrame(pd.read_csv(vec_file, header=0, encoding='utf-8-sig'))  # 无boom格式的utf-8
    data_list = []
    for items in df[which_seg]:
        data_list.append(eval(items))
    data = np.array(data_list)
    print('#' * 10 + ' load success..' + '#' * 10)
    return df['title'], data


def cosine_simi(seg):
    """
    计算得到相似度矩阵
    :return:
    """
    title, vec = load_vec(seg)
    dist_matrix = cosine_similarity(vec)
    return title, dist_matrix


def top_n_simi(sentence_id, count, title, dist_matrix):
    """
    :param sentence_id: 求相似度的句子id
    :param count: 展示多少个相似句子
    :param title: 标题列表
    :param dist_matrix: 相似度矩阵
    :return:
    """
    sentence = title[sentence_id]
    dist_title = dist_matrix[sentence_id, :]

    series = pd.Series(dist_title).sort_values(0, False)  # 降序排列

    print('=' * 20)
    print(sentence)
    print('='*20)
    for index in series.index[:count]:
        print(series[index], title[index])


if __name__ == '__main__':
    '''
    'baidu_seg_vec', 'tencent_seg_vec', 'ali_seg_vec'
    '''
    title, dist_matrix = cosine_simi('title_vec')
    top_n_simi(999, 20, title, dist_matrix)
    top_n_simi(888, 20, title, dist_matrix)
    top_n_simi(777, 20, title, dist_matrix)
    top_n_simi(8515, 20, title, dist_matrix)
    top_n_simi(723, 20, title, dist_matrix)
    top_n_simi(108, 20, title, dist_matrix)
    top_n_simi(8888, 20, title, dist_matrix)
    top_n_simi(1, 20, title, dist_matrix)
    top_n_simi(2, 20, title, dist_matrix)
    top_n_simi(3, 20, title, dist_matrix)
    top_n_simi(4, 20, title, dist_matrix)
    top_n_simi(5, 20, title, dist_matrix)
    top_n_simi(6, 20, title, dist_matrix)
    top_n_simi(7, 20, title, dist_matrix)
    top_n_simi(8, 20, title, dist_matrix)
    top_n_simi(9, 20, title, dist_matrix)
    top_n_simi(10, 20, title, dist_matrix)
    top_n_simi(11, 20, title, dist_matrix)
    top_n_simi(12, 20, title, dist_matrix)



