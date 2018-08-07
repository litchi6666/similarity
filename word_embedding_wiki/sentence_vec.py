import numpy as np
import pandas as pd
import jieba
from jieba.analyse.tfidf import TFIDF
from word_embedding_wiki.word_embedding import WordEmbedding
from sklearn.neighbors import NearestNeighbors


class SimiSentence:

    def __init__(self, N_neighbors = 20, sentence_vec_path='D:\\data\\videos\\title_vec.csv'):
        self.sentence_vec_path = sentence_vec_path
        self.n_neighbors = N_neighbors
        self.neigh = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto')
        # 使用kdtree搜索余弦相似距离
        self.file = pd.DataFrame(pd.read_csv(self.sentence_vec_path, header=0, encoding='utf-8-sig'))
        self.sen_vec = []
        for term in self.file['title_vec']:
            self.sen_vec.append(eval(term))
        self.neigh.fit(self.sen_vec)

        self.sentec_vec = SentenceVec()

    def get_simisSentence(self, sentence):
        title = self.sentec_vec.get_weight_sent_vec(sentence).tolist()
        neigh_dist, neigh_index = self.neigh.kneighbors(title)

        neigh_dist=neigh_dist.reshape(-1, 1)
        neigh_index=neigh_index.reshape(-1, 1)
        for i in range(self.n_neighbors):
            dist = (100 - neigh_dist[i]) / 100
            index = neigh_index[i]
            print(dist, self.file['title'][index])

    def get_simi(self, s1, s2):
        v1 = self.sentec_vec.get_weight_sent_vec(s1)
        v2 = self.sentec_vec.get_weight_sent_vec(s2)

        simi =np.sum(v1*v2)/(np.sqrt(sum(v1**2))*np.sqrt(sum(v2**2)))
        print(s1)
        print(s2)
        print('='*10,simi)

class SentenceVec:
    def __init__(self, sentence_vec_path='D:\\data\\videos\\title_vec.csv', model=WordEmbedding()):
        self.sentence_vec_path = sentence_vec_path
        self.model = model
        self.idf, self.median = TFIDF().idf_loader.get_idf()

    def get_tfidf(self, sentence):

        words = jieba.cut(sentence)
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0.0) + 1.0
        totle = sum(freq.values())
        tfidf_dict = {}
        for w in freq.keys():
            weight = self.idf.get(w, 0.) * freq[w] / totle
            tfidf_dict[w] = weight

        return tfidf_dict

    def get_sentence_vec(self, sentence):
        seg_list = self.model.__jieba_cut__(sentence)
        sen_vec = np.zeros(200)
        for word in seg_list:
            sen_vec += self.model.get_word_vec(word)
        return sen_vec/len(seg_list)

    def get_weight_sent_vec(self, sentence):
        '''
        带tfidf权重的词向量
        :param sentence:
        :return:
        '''
        tfidf_dict = self.get_tfidf(sentence)
        weight_totle = sum(tfidf_dict.values())

        sen_vec = np.zeros(200)
        for word in jieba.cut(sentence):
            sen_vec += self.model.get_word_vec(word) * tfidf_dict[word]

        if weight_totle > 0.0:
            return sen_vec / weight_totle
        else:
            return sen_vec

    def save_sentence_vec(self, sentence):
        sen_vec = pd.DataFrame(pd.read_csv(self.sentence_vec_path, header=0, encoding='utf-8-sig'))
        sen_vec['title'].append(sentence)
        sen_vec['title_vec'].append(self.get_sentence_vec(sentence))


def save_vec(weight=True):
    sv = SentenceVec()
    title_file = pd.DataFrame(pd.read_csv('D:\\data\\videos\\videos.csv', header=0, encoding='utf-8-sig'))
    # 生成词向量的标题文件
    title_list = []
    vec_list = []

    i = 0
    if not weight:
        for title in title_file['poi']:
            #title_list.append(title)
            # vec = sv.get_sentence_vec(title).tolist()
            # vec_list.append(vec)
            # i += 1
            # if i % 1000 == 0:
            #     print('已经计算%s行' % i)

            title_list.append(title_file['title'][i])
            vec = np.zeros(200)
            if title is not np.nan:
                words = title.split(',')
                for w in words:
                    vec += sv.model.get_word_vec(w)
            vec_list.append(vec.tolist())
            i += 1
            if i % 1000 == 0:
                print('已经计算%s行' % i)

    if weight:
        for title in title_file['video_name']:
            title_list.append(title)
            vec = sv.get_weight_sent_vec(title).tolist()
            vec_list.append(vec)
            i += 1
            if i % 1000 == 0:
                print('已经计算%s行' % i)

    vec_file = pd.DataFrame({'title': title_list, 'title_vec': vec_list})
    file = vec_file.to_csv('D:\\data\\videos\\videos_tag_vec.csv', index=False,
                           sep=',', encoding='utf-8', columns=['title', 'title_vec'])
    # 生成的词向量保存的位置
    print('saving file......')
    print('done!')


def simi_compute():
    sv = SimiSentence(100, sentence_vec_path='D:\\data\\videos\\videos_tag_vec.csv')
    # 'D:\\data\\textsimi\\BATresult_weight_vec.csv'
    sentences =[
        '广东KTV发生火灾致18死5伤 初步核查系人为纵火',
        '苏州惊现95后“卖鱼西施” 网友：连杀鱼都这么好看',
        '男女如何做才能同步“高”？',
        '理想的啪啪啪VS现实中的啪啪啪，未成年勿入！',
        '空姐遇害案告破 已完成DNA鉴定 打捞尸体确系杀害空姐嫌犯',
        '一男两女被困孤岛如何求生',
        '男生喜欢什么样的女生',
        '刘亦菲教你如何穿出天仙气质！快来看看吧',
        '中国最有可能合并的四座城市，合并后，将成为国际超级大都市！',
        '【脑补给】5张欺骗了许多人的爆红照片',
        '鸡肉新做法，在家用电饭煲做，上桌遭疯抢！',
        '这种红色100元，拿到了千万别花出去，银行正在紧急召回！',
    ]

    for sen in sentences:
        print('==' * 10 + sen)
        sv.get_simisSentence(sen)

    #sv.get_simi('鸡肉新做法，在家用电饭煲做，上桌遭疯抢！', '面粉开水一烫，一勺红糖一勺白糖，居然能做出这么好吃的零食')


if __name__ == '__main__':
    # save_vec(False)
    simi_compute()
