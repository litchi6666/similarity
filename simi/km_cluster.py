from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy as np
import pandas as pd


num_cluster = 20  # 聚多少个簇
vec_file = 'D:\\data\\textsimi\\matrix_title_vec.csv'  # 词向量文件
model_file = 'D:\\data\\textsimi\\cluster_model\\kmeans_result_10.pkl'  # 聚类模型保存的位置
seg = 'baidu_seg_vec'  # 使用的分词向量


def load_vec(which_seg):
    """
    加载标题+词向量
    :param which_seg: 使用的词向量
    :return: 标题，词向量
    """
    seg_list=['baidu_seg_vec', 'tencent_seg_vec', 'ali_seg_vec']  # video_name
    if which_seg not in seg_list:
        print('no such seg')
        return None
    print('#' * 10 + ' load data..' + '#' * 10)
    df = pd.DataFrame(pd.read_csv(vec_file, header=0, encoding='utf-8-sig'))  # 无boom格式的utf-8
    data_list = []
    for items in df['baidu_seg_vec']:
        data_list.append(eval(items))
    data = np.array(data_list)
    print('#' * 10 + ' load success..' + '#' * 10)
    return df['video_name'], data


def load_title_only():
    """
    加载标题文件
    :return:
    """
    df = pd.DataFrame(pd.read_csv(vec_file, header=0, encoding='utf-8-sig'))  # 无boom格式的utf-8
    print('标题加载完成')
    return df['video_name']


def km_cluster():
    """
    K-means聚类，并且保存模型
    :return:
    """
    title, vec = load_vec(seg)
    km = KMeans(n_clusters=num_cluster)
    km.fit(vec)
    joblib.dump(km, model_file)


def load_model():
    """
    加载聚类模型，显示聚类结果
    :return:
    """
    title = load_title_only()  # 加载标题
    km = joblib.load(model_file)
    cluster = km.labels_.tolist()  # 加载聚类结果

    title_cluster = {'title': title, 'cluster': cluster}
    video_cluster = pd.DataFrame(title_cluster)

    print('每个类别的个数')
    print(video_cluster['cluster'].value_counts())

    print(video_cluster.head())

    grouped = list(video_cluster.groupby(video_cluster['cluster']))

    for clus in grouped:
        print(clus)
        print('='*20)


if __name__ == '__main__':
    km_cluster()
    load_model()
