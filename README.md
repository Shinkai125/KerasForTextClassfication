# KerasForTextClassfication
中文Keras文本分类竞赛模板

### 词向量
### 预训练词向量精简 
抽取预训练的词向量前n行，只保留常用词汇，效果和这个 [项目](https://github.com/cliuxinxin/TX-WORD2VEC-SMALL)一样。
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