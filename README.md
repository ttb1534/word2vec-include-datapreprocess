# word2vec-include-datapreprocess
数据集链接：https://pan.baidu.com/s/1RHhHp8Y5_Y0AjYQ5Oa0CTA  提取码：1534

拿到数据集后处理思路如下：

<img src="https://gitee.com/ttb1534/typora-img-save/raw/master/image-20220306222246864.png" alt="image-20220306222246864" style="zoom:37%;" />

原始新闻数据集为news.txt，先进行数据预处理得到分词后的文件：
~~~python
python dataprocess.py
~~~~
得到cutdata.txt，再利用word2vec模型进行词嵌入：
~~~python
python train.py
~~~
得到word_embedding.txt文件，最后可测试效果：
~~~python 
python test.py
~~~

word2vec的处理思路如下：

<img src="https://gitee.com/ttb1534/typora-img-save/raw/master/image-20220306222748653.png" alt="image-20220306222748653" style="zoom:37%;" />

其主要思想和原理见：
