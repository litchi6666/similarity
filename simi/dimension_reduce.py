"""
降维查看数据的效果
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca():
    print('step1: load data..')
    vec_file = 'D:\\data\\textsimi\\matrix_title_vec.csv'
    df = pd.DataFrame(pd.read_csv(vec_file, header=0, ))
    data_list = []
    for items in df['baidu_seg_vec']:
        data_list.append(eval(items))
    data = np.array(data_list)
    print('step1 finish...')

    pca = PCA(n_components=2)
    reduce_data = pca.fit_transform(data)

    for i in range(len(reduce_data)):
        data_x = reduce_data[i][0]
        data_y = reduce_data[i][1]

    plt.scatter(data_x, data_y)
    plt.show()


def tsne():
    print('step1: load data..')
    vec_file = 'D:\\data\\textsimi\\matrix_title_vec.csv'
    df = pd.DataFrame(pd.read_csv(vec_file, header=0, ))
    data_list = []
    for items in df['baidu_seg_vec']:
        data_list.append(eval(items))
    data = np.array(data_list)
    print('step1 finish...')

    x_data = TSNE(n_components=2).fit_transform(data)

    plt.figure(figsize=(10, 5))
    plt.scatter(x_data[:, 0], x_data[:, 1])
    plt.show()


if __name__ == '__main__':
    #tsne()
    pca()
