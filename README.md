# KerasForTextClassfication
中文Keras文本分类竞赛模板

### 词向量
### 预训练词向量精简 
抽取预训练的词向量前n行(这是里腾讯开源的[词向量](https://ai.tencent.com/ailab/nlp/en/data/Tencent_AILab_ChineseEmbedding.tar.gz))，只保留常用词汇，效果和这个 [项目](https://github.com/cliuxinxin/TX-WORD2VEC-SMALL)一样。
```python
from tqdm import  tqdm
from embeddings import get_embeddings_index, build_embedding_weights
def convert_big_embeddings_to_small(big_filename, small_filename, number_line=500000):
    line_count = 0
    with open(big_filename) as reader, open(small_filename, 'w') as writer:
        for index, line in enumerate(tqdm(reader)):
            if line_count <= number_line:
                writer.write(line)
                line_count += 1
            else:
                break
    writer.close()
# 抽取预训练的词向量前n行，只保留常用词汇
convert_big_embeddings_to_small(big_filename='Tencent_AILab_ChineseEmbedding.txt',
                                small_filename='Tencent_AILab_ChineseEmbedding_1000000.txt',
                                number_line=1000000)
# 加载词向量
word_index = {'<PAD>': 0, '我': 1, '爱': 2, '你': 3}
embed_index = get_embeddings_index(embedding_file_or_type='./Tencent_AILab_ChineseEmbedding_500000.txt',
                                       cache_dir='.')
embed_weights, oov = build_embedding_weights(word_index, embed_index, return_oov=True)
print(embed_weights.shape) #词向量矩阵的shape
print(oov) # 打印超出词表的词汇
```

### 训练日志
```shell
2020-12-09 06:35:20,880 - train.py[line:57] - INFO: Reading dataset from csv file:  chnsenticorp/train.tsv
   label                                             text_a
0      1  选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全...
1      1  15.4寸笔记本的键盘确实爽，基本跟台式机差不多了，蛮喜欢数字小键盘，输数字特方便，样子也很...
2      0                               房间太小。其他的都一般。。。。。。。。。
3      0  1.接电源没有几分钟,电源适配器热的不行. 2.摄像头用不起来. 3.机盖的钢琴漆，手不能摸...
4      1  今天才知道这书还有第6卷,真有点郁闷:为什么同一套书有两种版本呢?当当网是不是该跟出版社商量...
2020-12-09 06:35:20,926 - train.py[line:61] - INFO: Cleaning texts
2020-12-09 06:35:20,952 - train.py[line:63] - INFO: Tokenizing texts
100%|█████████████████████████████████████| 9146/9146 [00:02<00:00, 3284.13it/s]
2020-12-09 06:35:23,739 - train.py[line:57] - INFO: Reading dataset from csv file:  chnsenticorp/dev.tsv
   label                                             text_a
0      1  這間酒店環境和服務態度亦算不錯,但房間空間太小~~不宣容納太大件行李~~且房間格調還可以~~...
1      1  <荐书> 推荐所有喜欢<红楼>的红迷们一定要收藏这本书,要知道当年我听说这本书的时候花很长时...
2      0     商品的不足暂时还没发现，京东的订单处理速度实在.......周二就打包完成，周五才发货...
3      1    ２００１年来福州就住在这里，这次感觉房间就了点，温泉水还是有的．总的来说很满意．早餐简单了些．
4      1  不错的上网本，外形很漂亮，操作系统应该是个很大的 卖点，电池还可以。整体上讲，作为一个上网本...
2020-12-09 06:35:23,750 - train.py[line:61] - INFO: Cleaning texts
2020-12-09 06:35:23,754 - train.py[line:63] - INFO: Tokenizing texts
100%|█████████████████████████████████████| 1200/1200 [00:00<00:00, 4792.29it/s]
2020-12-09 06:35:24,005 - train.py[line:57] - INFO: Reading dataset from csv file:  chnsenticorp/test.tsv
   label                                             text_a
0      1                         这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般
1      0  怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片！开始还怀疑是...
2      0  还稍微重了点，可能是硬盘大的原故，还要再轻半斤就好了。其他要进一步验证。贴的几种膜气泡较多，...
3      1                              交通方便；环境很好；服务态度很好 房间较小
4      1  不错，作者的观点很颠覆目前中国父母的教育方式，其实古人们对于教育已经有了很系统的体系了，可是...
2020-12-09 06:35:24,015 - train.py[line:61] - INFO: Cleaning texts
2020-12-09 06:35:24,019 - train.py[line:63] - INFO: Tokenizing texts
100%|█████████████████████████████████████| 1200/1200 [00:00<00:00, 4790.16it/s]
2020-12-09 06:35:24,270 - train.py[line:81] - INFO: Fitting tokenizer...
2020-12-09 06:35:24,971 - train.py[line:86] - INFO: Building training set...
2020-12-09 06:35:25,403 - train.py[line:90] - INFO: Building validation set...
2020-12-09 06:35:25,458 - train.py[line:94] - INFO: Building test set ...
2020-12-09 06:35:25,511 - train.py[line:97] - INFO: Padding sequences...
Downloading data from http://212.129.155.247/embedding/sgns.literature.word.txt.zip
184459264/184454252 [==============================] - 50s 0us/step
2020-12-09 06:36:19,511 - embeddings.py[line:56] - INFO: Building embeddings index...
building embeddings index: 187960it [00:13, 13521.81it/s]
2020-12-09 06:36:33,412 - embeddings.py[line:73] - INFO: Loading embeddings for all words in the corpus
2020-12-09 06:36:33,419 - embeddings.py[line:76] - INFO: Embedding dimensions: 300
2020-12-09 06:36:33,466 - embeddings.py[line:86] - INFO: Percentage of tokens in pretrained embeddings: 70.48386681430742%
['携程', '一本', '第一次', '一次', '这是', '书中', '第二天', '两个', '一天', '一种', '几个', '显卡', '四星', '不太', 'VISTA', '当当网', '一套', 'vista', '还会', '一张', '一看', '几天', '预装', '华硕', '一句', '一位', '标间', '两天', '一间', '有个', '触摸板', '1G', '2G', '一股', 'Vista', '这点', '几次', '三个', '很近', '一条', '一遍', '书里', '发热量', '这书', '我要', '半个', '一晚', '酒店设施', '几页', '我订']
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 1024)]       0                                            
__________________________________________________________________________________________________
embedding (Embedding)           (None, 1024, 300)    11650200    input_1[0][0]                    
__________________________________________________________________________________________________
spatial_dropout1d (SpatialDropo (None, 1024, 300)    0           embedding[0][0]                  
__________________________________________________________________________________________________
bidirectional (Bidirectional)   (None, 1024, 512)    1140736     spatial_dropout1d[0][0]          
__________________________________________________________________________________________________
bidirectional_1 (Bidirectional) (None, 1024, 256)    656384      bidirectional[0][0]              
__________________________________________________________________________________________________
attention (Attention)           (None, 256)          1280        bidirectional_1[0][0]            
__________________________________________________________________________________________________
global_average_pooling1d (Globa (None, 256)          0           bidirectional_1[0][0]            
__________________________________________________________________________________________________
global_max_pooling1d (GlobalMax (None, 256)          0           bidirectional_1[0][0]            
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 768)          0           attention[0][0]                  
                                                                 global_average_pooling1d[0][0]   
                                                                 global_max_pooling1d[0][0]       
__________________________________________________________________________________________________
dense (Dense)                   (None, 512)          393728      concatenate[0][0]                
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 128)          65664       dense[0][0]                      
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 2)            258         dense_1[0][0]                    
==================================================================================================
Total params: 13,908,250
Trainable params: 2,258,050
Non-trainable params: 11,650,200
__________________________________________________________________________________________________
Epoch 1/4
115/115 [==============================] - 28s 244ms/step - loss: 0.5714 - acc: 0.6938 - val_loss: 0.3907 - val_acc: 0.8306 - lr: 0.0010
Epoch 2/4
115/115 [==============================] - 27s 238ms/step - loss: 0.4327 - acc: 0.8111 - val_loss: 0.3846 - val_acc: 0.8372 - lr: 0.0010
Epoch 3/4
115/115 [==============================] - 28s 239ms/step - loss: 0.3827 - acc: 0.8387 - val_loss: 0.3685 - val_acc: 0.8672 - lr: 0.0010
Epoch 4/4
115/115 [==============================] - 27s 239ms/step - loss: 0.3495 - acc: 0.8555 - val_loss: 0.3116 - val_acc: 0.8694 - lr: 0.0010
38/38 [==============================] - 2s 62ms/step - loss: 0.3399 - acc: 0.8600
acc: 86.00%
```