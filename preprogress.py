"""
@file: preprogress.py
@time: 2020-12-02 15:52:58
"""
import logging
import re
import string

import jieba_fast as jieba
from zhon import hanzi

jieba.setLogLevel(logging.INFO)

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',
          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',
          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']


# 定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line


# 空格分隔标点符号
def clean_punctuation(x):
    x = str(x)
    for punct in puncts + list(string.punctuation) + list(hanzi.punctuation):
        if punct in x:
            x = x.replace(punct, f' {punct} ')
    return x


# 清理数字
def clean_numbers(x):
    return re.sub(r'\d+', ' ', x)


# 清理HTML Tag
def clean_html_tags(sentence):
    """
    Function to remove a sentence having HTML tags
    :param sentence: string
    :return: string
    """
    clean_text = re.sub('<[^<]+?>', '', sentence).replace('\n', '').strip()
    return clean_text


# 清理url
def clean_urls(sentence):
    sentence = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', sentence,
                      flags=re.MULTILINE)
    return sentence


# 清理特殊字符，可根据需要添加
def clean_special_chars(text, specials=None):
    if specials is None:
        specials = {'\u2003': ''}
    for s in specials:
        text = text.replace(s, specials[s])
    return text


# 清理文本，汇总
def clean_text(sentence):
    sentence = clean_urls(sentence)
    sentence = clean_numbers(sentence)
    sentence = clean_punctuation(sentence)
    sentence = clean_html_tags(sentence)
    sentence = clean_special_chars(sentence)
    return sentence


if __name__ == '__main__':
    doc = """又是一年落叶黄，一阵秋雨一阵凉；整日奔波工作忙，出门别忘添衣裳。金秋时节，正值装修旺季，集美家居继续带消费者们“乘风破浪”。为满足消费者装修置家需求，帮助消费者选购到质优价美的建材家居用品，集美家居北苑商场将于9月10日-13日举办金秋爆破团购会活动。   活动期间，全年最低折扣、满减满赠、幸运抽奖、9元秒家具等实实在在的优惠福利让消费者拿到手软。据活动相关负责人介绍，本次团购会将是集美家居北苑商场年度内优惠力度最大的一次促销活动，可以说是一场不容错过的家居盛“惠”。  具体优惠福利如下：  （一）各大品牌推出全年最低折扣回馈消费者；  （二）集美家居北苑商场推出满1000元减100元优惠券；  （三）消费者可参与抢购15元升300元、50元升1000元升值券；  （四）此外，还有满赠家居大礼包，幸运大抽奖，9元秒家具等丰富多彩的活动等候消费者参与。  集美家居北苑商场坐落在北五环的朝阳区红军营南路19号，临近地铁5号线北苑路北站，附近有多条公交线路，交通便利；集美家居北苑商场内设有大型停车场，便于驱车前来购物的消费者享受停车服务；另有班车预约服务供消费者享受，随叫随到。  集美家居北苑商场定位于京北地区现代化、智能化、体验化、品牌化的一站式大家居商场。一直以来，集美家居北苑商场坚持以诚信赢得顾客，多年被北京市工商局评为“诚信经营示范市场”“消费者绿色通道”。  据了解，疫情期间集美家居北苑商场进行了全面升级改造，提供品类齐全的商品、购物无忧的售后服务，使购物环境更加舒适、健康、温馨，以便消费者逛得舒心、放心、省心。  集美家居北苑商场将带领全体员工真诚欢迎新老朋友的光临，并竭诚服务好每一位到店的消费者。  选择集美家居，就是选择美好生活！原文网址:神兽归笼日、装修正当时——集美家居北苑商场金秋爆破团购会即将启动http://www.jiaju82.com/news-view-id-720242.html"""
    print(clean_text(doc))
    print(jieba.lcut(clean_text(doc)))
