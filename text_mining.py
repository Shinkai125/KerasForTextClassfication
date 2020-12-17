"""
@file: text_mining.py
@time: 2020-12-09 17:23:42
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm




def label_distribution(df):
    d = {'label': df.value_counts().index, 'count': df.value_counts()}
    df_cat = pd.DataFrame(data=d).reset_index(drop=True)
     
    myfont = fm.FontProperties(fname='SimHei.ttf')  # 设置字体
    df_cat.plot(x='label', y='count', kind='bar', legend=False, figsize=(8, 5))
    plt.title("类目数量分布", fontproperties=myfont)
    plt.ylabel('数量', fontproperties=myfont, fontsize=18)
    plt.xlabel('类目', fontproperties=myfont, fontsize=18)
    plt.show()


if __name__ == '__main__':
    train_data = pd.read_csv('chnsenticorp/train.tsv', sep='\t')
    label_distribution(train_data['label'])
