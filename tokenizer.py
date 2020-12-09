import logging
import jieba_fast as jieba
from pprint import pprint

from tqdm import tqdm

jieba.setLogLevel(logging.INFO)


def tokenize(texts, sep=',', join=False):
    tokenize_texts = []
    if join:
        for text in tqdm(texts):
            try:
                cut_text = jieba.lcut(text)
            except:
                print(text)
                exit(0)
            join_text = sep.join(cut_text)
            tokenize_texts.append(join_text)
        return tokenize_texts
    else:
        return [jieba.lcut(text) for text in texts]


if __name__ == '__main__':
    document_list = ["行政机关强行解除行政协议造成损失，如何索取赔偿？",
                     "借钱给朋友到期不还得什么时候可以起诉？怎么起诉？",
                     "我在微信上被骗了，请问被骗多少钱才可以立案？",
                     "公民对于选举委员会对选民的资格申诉的处理决定不服，能不能去法院起诉吗？",
                     "有人走私两万元，怎么处置他？",
                     "法律上餐具、饮具集中消毒服务单位的责任是不是对消毒餐具、饮具进行检验？"]

    pprint(tokenize(document_list))
