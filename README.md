# word2vec-include-datapreprocess
数据集链接：https://pan.baidu.com/s/1RHhHp8Y5_Y0AjYQ5Oa0CTA  提取码：1534

拿到数据集后处理思路如下：

<div align=center><img src="https://pic2.zhimg.com/80/v2-6ebe31b877d20e425bd2ce057b00f421_1440w.webp" alt="image-20220306222246864" width="700px" /></div>

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

在上文提供的数据集链接中的datasave文件夹已经包含经处理过后的cutdata_prepare.txt和word_embedding_pretrained.txt，可直接用来测试

word2vec的处理思路如下：

<div align=center><img src="https://pic2.zhimg.com/80/v2-9aff3901dfbd18de347951df5e599979_1440w.webp" alt="image-20220306222748653" width="700px" /></div>

其主要思想和原理见：https://zhuanlan.zhihu.com/p/476920885
