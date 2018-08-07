import pandas as pd
from gensim.models.word2vec import Word2Vec

# corpus_after_jieba = 'D:\\data\\textsimi\\jieba_w2c\\tok_result.csv'
# w2c_model = 'D:\\data\\textsimi\\jieba_w2c\\w2c.model'

corpus_after_jieba = 'D:\\data\\videos\\tok_result.csv'
w2c_model = 'D:\\data\\videos\\w2c.model'


def train_model():
    crops = pd.read_csv(corpus_after_jieba, header=0, encoding='utf-8-sig')['seg']
    sentence = []
    for term in crops:
        sentence.append(term.split(','))
    model = Word2Vec(sentence,
                     size=200,
                     window=1,
                     min_count=2,
                     workers=2,
                     sg=0)
    model.save(w2c_model)
    # for i in model.vocab.keys(): #vocab是dict
    #     print type(i)
    #     print i
    # model = Word2Vec.load('word2vec_model')


def load_model():
    model = Word2Vec.load(w2c_model)
    print('load success')
    print(model.most_similar('男孩'))
    print(model.wv['小哥'].shape)


train_model()
load_model()