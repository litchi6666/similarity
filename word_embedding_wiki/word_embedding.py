import numpy as np
import jieba

import gensim
from gensim.models.word2vec import Word2Vec
import re


class WordEmbedding:
    def __init__(self, model_path = 'D:\\data\\wiki_embedding\\model\\wiki.zh.model'):
        self.model_path = model_path
        self.model = Word2Vec.load(model_path)
        print('=' * 20 + 'model load success' + '=' * 20)

    def get_word_vec(self, word):
        r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~，。、【 】 “”：；（）《》‘’{}？！()、%^>℃：.”“^-——=&#@￥' \
            '1234567890]+'
        word = re.sub(r, '', word)
        if word in self.model:
            return self.model[word]
        else:
            return np.zeros(200)

    def __jieba_cut__(self, sentence):
        cut_result = jieba.cut(sentence)
        list_cut = []
        for word in cut_result:
            list_cut.append(word)
        #print(list_cut)
        return list_cut

    def __n_similarity__(self, ws1, ws2):
        """Compute cosine similarity between two sets of words.

        Parameters
        ----------
        ws1 : list of str
            Sequence of words.
        ws2: list of str
            Sequence of words.

        Returns
        -------
        numpy.ndarray
            Similarities between `ws1` and `ws2`.

        """
        if not(len(ws1) and len(ws2)):
            raise ZeroDivisionError('At least one of the passed list is empty.')
        v1 = [self.get_word_vec(word) for word in ws1]
        v2 = [self.get_word_vec(word) for word in ws2]
        return np.dot(gensim.matutils.unitvec(np.array(v1).mean(axis=0)), gensim.matutils.unitvec(np.array(v2).mean(axis=0)))

    def sentence_simi(self, s1, s2):
        list1 = self.__jieba_cut__(s1)
        list2 = self.__jieba_cut__(s2)
        return self.__n_similarity__(list1, list2)



if __name__ == '__main__':
    we = WordEmbedding()
    print(we.sentence_simi('男生喜欢什么样的女生', '水瓶座女生喜欢什么样的男生? 为什么你总是追不到她?'))  # 0.84

