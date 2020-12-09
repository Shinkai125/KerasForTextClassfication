# KerasForTextClassfication
中文Keras文本分类竞赛模板

### 词向量
### 抽取预训练的词向量前n行，只保留常用词汇，效果和这个 [项目](https://github.com/cliuxinxin/TX-WORD2VEC-SMALL)一样。
```python
from tqdm import  tqdm
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

convert_big_embeddings_to_small(big_filename='Tencent_AILab_ChineseEmbedding.txt',
                                small_filename='Tencent_AILab_ChineseEmbedding_1000000.txt',
                                number_line=1000000)
```