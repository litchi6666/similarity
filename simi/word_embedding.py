"""
从数据库中获取每个词的词向量，将词向量保存到dict中并返回

"""

import numpy as np
import pymysql


def get_embeding_from_mysql():
    print('start get word embeding,')

    db = pymysql.connect(host="localhost", user="root", password='1234', db="batresult", charset='utf8')
    cursor = db.cursor()
    sql = "SELECT baidu_seg,baidu_vec FROM matrix_baidu_seg"
    wordemb_dict = {}
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
        for row in result:
            # 某些词向量没有生成为[]的形式不导入
            if len(row[1]) > 3:
                word = row[0]
                embedding = np.array(eval(row[1]))
                wordemb_dict[word] = embedding
    except:
        print("woring!",word,embedding)

    db.close()
    print("got it!")
    return wordemb_dict


if __name__ == "__main__":
    vec_dict = get_embeding_from_mysql()
    print(len(vec_dict))
    print(vec_dict['小伙'].shape)
