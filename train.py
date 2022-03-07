"""
This code is modified from Adoni's repository.
https://github.com/Adoni/word2vec_pytorch

"""
from torch.autograd import Variable
import torch
from dataprocess import CutDataProcess
from model import SkipGramModel
import torch.optim as optim
from tqdm import tqdm

inputdata_path = r'.\dataset\news.txt'
stopword_path = r'.\dataset\cn_stopwords.txt'
datasave_path = r'.\datasave\cutdata0.txt'
cutdata_path = datasave_path
output_path = r'.\datasave\word_embedding.txt'

class Word2Vec:
    def __init__(self,
                 input_file_name,
                 output_file_name,
                 emb_dimension=100,
                 batch_size=300,
                 window_size=5,
                 neg_count=5,
                 iteration=1,
                 initial_lr=0.005,
                 min_count=5):
        """
        Args:
            input_file_name: Name of the cutdata file. Each line is a sentence splited with space.
            output_file_name: Name of the final embedding file.
            emb_dimention: Embedding dimention, typically from 50 to 500.
            batch_size: The count of word pairs for one forward.
            window_size: Max skip length between words.
            neg_count: The num of negative samples correspond to a positive sample
            iteration: Controlling the number of training iterations.
            initial_lr: Initial learning rate.
            min_count: The minimal word frequency, words with lower frequency will be filtered.
        """
        self.data = CutDataProcess(input_file_name, min_count)
        self.output_file_name = output_file_name
        self.iteration = iteration
        self.emb_size = len(self.data.word_id)
        self.emb_dimension = emb_dimension
        self.neg_count = neg_count
        self.batch_size = batch_size
        self.window_size = window_size
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.device = ("cuda" if torch.cuda.is_available else "cpu")
        if self.device=="cuda":
            self.skip_gram_model.to(self.device)
        self.optimizer = optim.SGD(
            self.skip_gram_model.parameters(), lr=self.initial_lr)

    def train(self):
        pair_count = self.data.countpair(self.window_size)
        batch_count = self.iteration*pair_count/self.batch_size
        process_bar = tqdm(range(int(batch_count))) 
        for i in process_bar:
            pos_pairs = self.data.getpospair(self.batch_size,
                                                  self.window_size)
            neg_v = self.data.getnegv(pos_pairs,self.neg_count)
            pos_u = [pair[0] for pair in pos_pairs] # 取出中心词
            pos_v = [pair[1] for pair in pos_pairs] # 取出背景词
            
            pos_u = Variable(torch.LongTensor(pos_u)).to(self.device)
            pos_v = Variable(torch.LongTensor(pos_v)).to(self.device)
            neg_v = Variable(torch.LongTensor(neg_v)).to(self.device)

            self.optimizer.zero_grad()
            loss = self.skip_gram_model.forward(pos_u,pos_v,neg_v)
            loss.backward()
            self.optimizer.step()

            process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                        (loss.item(),
                                         self.optimizer.param_groups[0]['lr']))
            if i*self.batch_size%100000==0:
                lr = self.initial_lr*(1-1*i/batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        self.skip_gram_model.savemodel(
            self.data.id_word, self.output_file_name)


if __name__ == '__main__':
    # 数据预处理->分割成词
    # data = DataPreprocess(inputdata_path,stopword_path)
    # data.savecutdata(datasave_path)
    word2vec = Word2Vec(cutdata_path,output_path)
    word2vec.train()