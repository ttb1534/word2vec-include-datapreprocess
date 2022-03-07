"""
This code is modified from Adoni's repository.
https://github.com/Adoni/word2vec_pytorch

"""
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    def __init__(self,emb_length,emb_dimension):
        super(SkipGramModel,self).__init__()
        self.emb_length = emb_length
        self.emb_dimension = emb_dimension
        self.u_embedding = nn.Embedding(self.emb_length,self.emb_dimension,sparse=True)
        self.v_embedding = nn.Embedding(self.emb_length,self.emb_dimension,sparse=True)
        initrange = 0.5/self.emb_dimension
        self.u_embedding.weight.data.uniform_(-initrange,initrange)
        self.v_embedding.weight.data.uniform_(-0,0)
    
    def forward(self,pos_u,pos_v,neg_v):
        pos_embu = self.u_embedding(pos_u)
        pos_embv = self.v_embedding(pos_v)
        neg_embv = self.v_embedding(neg_v)
        # 逐元素相乘计算正样本对相似度,log以e为底
        pos_score = torch.mul(pos_embu,pos_embv)
        pos_score = F.logsigmoid(torch.sum(pos_score,dim=1).squeeze())
        # 计算负样本相似度，neg_embv.shape=[batch_size,neg_count,emb_dimension]
        neg_score = torch.matmul(neg_embv,pos_embu.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1*neg_score)
        return -1*(torch.sum(neg_score)+torch.sum(pos_score))
        
    def savemodel(self,id_word,outdata_path):
        if torch.cuda.is_available:
            # numpy无法直接从CUDA中读取数据，需要先将CUDA中的数据转移到cpu中
            embedding = self.u_embedding.weight.data.cpu().numpy()
        else:
            embedding = self.u_embedding.weight.data.numpy()
        with open(outdata_path,'w',encoding='UTF-8') as f:
            f.write('词语数量: {}    嵌入维度: {}\n'.format(len(id_word),self.emb_dimension))
            for i,word in id_word.items():
                e = embedding[i]
                e = ' '.join(map((lambda x:str(x)),e))
                f.write('{} {}\n'.format(word,e))
                