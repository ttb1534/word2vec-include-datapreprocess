# data preprocess
#-*- coding : utf-8-*-
# coding:unicode_escape
import jieba
import pandas as pd
from sklearn import preprocessing
import re
import numpy as np
from collections import deque
import operator

np.random.seed(0)


class DataPreprocess():
    def __init__(self,inputdata_path,stopword_path):
        self.dataed,self.labeled = self.dataload(inputdata_path,stopword_path)

    def dataload(self,inputdata_path,stopword_path):
        train_data = pd.read_csv(inputdata_path,delimiter='\t',
                                 lineterminator='\n',encoding='UTF-8',header=None)
        rawlabellist = []
        datalist = [[] for _ in range(train_data.shape[0])]
        for i in range(train_data.shape[0]):
            rawlabellist.append(train_data.iloc[i,0])
        labellist = preprocessing.LabelEncoder().fit_transform(rawlabellist) 
        for i in range(train_data.shape[0]):
            datalist[i] = train_data.iloc[i,1] 
        with open(stopword_path,'r',encoding="UTF-8") as f:
            self.stopword = []
            for line in f.readlines():
                self.stopword.append(line.strip())
        return datalist,labellist
        
    def stopworddelete(self,sentence):
        word_num = 0
        while(word_num<len(sentence)):
            if sentence[word_num] in self.stopword:
                sentence.remove(sentence[word_num])
                word_num -= 1
            word_num += 1
        return sentence   
    
    def wordcut(self,sentence):
        criterion = r'[\u4e00-\u9fa5a-zA-Z]+' # 文本正则化，只留下汉字和英文
        sentence = re.findall(criterion,sentence)
        sentence = "".join(sentence) # 拼接列表为字符串
        sentence = jieba.lcut(sentence) # jieba分词
        sentence = self.stopworddelete(sentence) # 去停止词
        return sentence     
        
    def savecutdata(self,datasave_path):
        with open(datasave_path,"w",encoding="UTF-8") as f:
            for i in range(len(self.dataed)):
                cut_words = self.wordcut(self.dataed[i])
                combin_word = ""
                for i in cut_words:
                    combin_word += i
                    combin_word += " "
                f.write(combin_word)
                f.write('\n')
                
class CutDataProcess:
    def __init__(self,cutdata_path,min_count):
        self.sentence_index = 0
        self.pos_pair = deque()
        self.min_count = min_count
        self.dataprocess(cutdata_path)
            
    def dataprocess(self,cutdata_path):
        cut_data = pd.read_csv(cutdata_path,header=None,lineterminator='\n',encoding='UTF-8',delimiter='\t')
        self.cut_list = []
        for i in range(cut_data.shape[0]):
            self.cut_list.append(cut_data.iloc[i][0].strip().split())
        word_frequency = dict()
        for i in self.cut_list: # 统计词频
            for j in i:
                try:
                    word_frequency[j] += 1
                except:
                    word_frequency[j] = 1
        self.word_id = dict()
        self.id_word = dict()
        self.word_frequency = []
        self.highword_frequency = dict()
        word_index = 0
        for word,count in word_frequency.items(): 
            if count < self.min_count:
                continue
            self.word_id[word] = word_index
            self.id_word[word_index] = word
            self.highword_frequency[word] = count
            word_index += 1
        self.word_count = len(self.word_id)
        self.sentence_count = len(self.cut_list)
        # 按词频重新排列字典,sorted输出为元组格式
        self.word_frequency = sorted(self.highword_frequency.items(),key=lambda x:x[1])
        self.allword_num = 0
        for item in self.highword_frequency.items():
            self.allword_num += item[1]

    def countpair(self,window_size):
        # 粗略估计正样本对的个数
        return self.allword_num*(2*window_size-1)-window_size*(
            window_size+1)*self.sentence_count

    def getpospair(self,batch_size,window_size):
        while len(self.pos_pair)<batch_size:
            try:
                sentence = self.cut_list[self.sentence_index]
            except: # iteration>1时要再次遍历数据
                self.sentence_index = 0
                sentence = self.cut_list[self.sentence_index]
            self.sentence_index += 1
            posword_id = []
            for word in sentence:
                try:
                    if word in self.highword_frequency:
                        posword_id.append(self.word_id[word])
                except:
                    continue
            for i,u in enumerate(posword_id): # i为中心词索引，u为中心词
                for j,v in enumerate(posword_id[max(i-window_size, 0):min(i+window_size+1,len(posword_id))]):
                    if i==j:
                        continue
                    self.pos_pair.append((u,v))

        posbatch_pair = []
        for _ in range(batch_size):
            posbatch_pair.append(self.pos_pair.popleft())
        return posbatch_pair
              
    def getnegv(self,posbatch_pair,neg_count):
        # 负采样频率设置为词频率的0.75次方,提高低频词的选中概率(只有数组元素可以在内部进行计算)
        sam_frequency = np.array(list(self.highword_frequency.values()))**0.75
        # 使用排序过后的词典选词(便于看清选词概率)
        # sam_frequency = np.array(list(dict(self.word_frequency).values()))**0.75
        # sam_probability是一个数组，里面是所有词预计被取为负样本的概率
        sam_probability = sam_frequency/sum(sam_frequency)
        maxpro_index,_ = max(enumerate(sam_probability),key=operator.itemgetter(1))
        if sum(sam_probability)>1:
            sam_probability[maxpro_index] = sum(sam_probability)-1+sam_probability[maxpro_index]
        else:
            sam_probability[maxpro_index] = 1-sum(sam_probability)+sam_probability[maxpro_index]
        # 按词频来挑选负样本
        neg_v = np.random.choice(range(len(sam_probability)),replace=True,size=(len(posbatch_pair),neg_count),p=sam_probability).tolist()
        return neg_v

# 处理原始数据，将数据分词、去停用词并保存为cutdata.txt为词嵌入做准备               
if __name__ == '__main__':
    inputdata_path = r'.\dataset\news.txt'
    stopword_path = r'.\dataset\cn_stopwords.txt'
    datasave_path = r'.\datasave\cutdata0.txt'
    data = DataPreprocess(inputdata_path,stopword_path)
    data.savecutdata(datasave_path)