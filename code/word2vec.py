# first import the packages we need
import collections
import math
import random
import sys
import time
import os
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import joblib
import torch.nn as nn
from tqdm import tqdm

class WVCY():
    def __init__(self,n = None):
        # next we need to read the raw data
        self.raw_dataset = joblib.load("/root/autodl-tmp/PAPER/DATA/RAW_DATA/important/SentenceMatchData_kbs_cw.pkl")[["stock_code","date","word"]]
        if n:
            self.raw_dataset = self.raw_dataset.sample(n)
        self.raw_dataset = self.raw_dataset.drop_duplicates("word")
        self.raw_dataset["word"] = self.raw_dataset["word"].apply(lambda x:x.split(" "))
        self.raw_dataset["date"] = self.raw_dataset["date"].apply(lambda x:str(int(x)))
        self.raw_dataset["date"] = self.raw_dataset["date"].apply(lambda x:x[:4] + str(int(x[4:6]) // 6))
        self.date_list = self.raw_dataset["date"].unique().tolist()
        self.stock_code_list = self.raw_dataset["stock_code"].unique().tolist()
        self.raw_dataset = self.raw_dataset.reset_index(drop = True)
        self.raw_dataset = self.raw_dataset.to_dict()

        # first we need to give the word in sentence a token
        self.counter = collections.Counter([tk for _,st in self.raw_dataset["word"].items() for tk in st])
        self.counter = dict(filter(lambda x:x[1] >= 5,self.counter.items()))

        # secind we need to add year and company code into the list, and each code and year will have a token too
        self.idx2word = [tk for tk,_ in self.counter.items()] + self.date_list + self.stock_code_list
        self.word2idx = {tk:idx for idx,tk in enumerate(self.idx2word)}
        self.word_dataset = [[self.word2idx[tk] for tk in st if tk in self.word2idx] for _,st in self.raw_dataset["word"].items()]
        self.stock_code_dataset = [[self.word2idx[code]] for _,code in self.raw_dataset["stock_code"].items()]
        self.date_dataset = [[self.word2idx[d]] for _,d in self.raw_dataset["date"].items()]
        #dataset = [self.word_dataset[i] + self.stock_code_dataset[i] + self.date_dataset[i] for i in range(len(self.stock_code_dataset))]

        # 二次采样，即数据集中每个索引词都有一定几率被舍弃
        self.word_num = sum([len(st) for st in self.word_dataset])
    def subsample(self,idx,t = 1e-4):
        return random.uniform(0,1) < 1 - math.sqrt(t * self.word_num / self.counter[self.idx2word[idx]])

    @property
    def subsample_dataset(self):
        subsample_dataset = [[tk for tk in st if not self.subsample(tk)] for st in self.word_dataset]
        return subsample_dataset

    # 提取中心词和背景词
    # 我们将与中心词距离不超过背景窗口大小的词作为它的背景词。下面定义函数提取出所有中心词和它们的背景词。它每次在整数1和max_window_size（最大背景窗口）之间随机均匀采样一个整数作为背景窗口大小
    
    def get_centers_and_contexts(self,dataset,max_window_size = 5):
        centers, contexts = [], []
        for i in range(len(dataset)):
            st = dataset[i]
            if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
                continue
            centers += st
            for center_i in range(len(st)):
                window_size = random.randint(1, max_window_size)
                indices = list(range(max(0, center_i - window_size),
                                    min(len(st), center_i + 1 + window_size)))
                indices.remove(center_i)  # 将中心词排除在背景词之外
                # 在背景词中加入代码和日期索引
                contexts.append([st[idx] for idx in indices] + self.stock_code_dataset[i] + self.date_dataset[i])
        return centers, contexts

    @property
    def centers_contexts(self):
        all_centers, all_contexts = self.get_centers_and_contexts(self.subsample_dataset)
        return all_centers,all_contexts

    # 负采样
    # 我们使用负采样来进行近似训练。对于一对中心词和背景词，我们随机采样K个噪声词（实验中设K=5）。根据word2vec论文的建议，噪声词采样概率P(w)设为w词频与总词频之比的0.75次方
    def get_negatives(self,all_contexts, all_centers, sampling_weights, K):
        all_negatives, neg_candidates, i = [], [], 0
        population = list(range(len(sampling_weights)))
        for j in range(len(all_contexts)):
            contexts = all_contexts[j]
            center_j = all_centers[j]
            negatives = []
            while len(negatives) < len(contexts) * K:
                # 当neg_candidates为空或者已经用过时
                if i == len(neg_candidates):
                    # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                    # 为了高效计算，可以将k设得稍大一点
                    i, neg_candidates = 0, random.choices(
                        population, sampling_weights, k=int(1e5))
                neg, i = neg_candidates[i], i + 1
                # 噪声词不能是背景词和中心词
                if neg not in set(contexts) and neg != center_j:
                    negatives.append(neg)
            all_negatives.append(negatives)
        return all_negatives

    @property
    def centers_contexts_nagatives(self):
        sampling_weights = [self.counter[w]**0.75 for w in self.counter.keys()]
        all_centers,all_contexts = self.centers_contexts
        all_negatives = self.get_negatives(all_contexts, all_centers, sampling_weights, 5)
        return all_centers,all_contexts,all_negatives


class MyDataset(Dataset):
    def __init__(self,centers,contexts,negatives):
        assert len(centers) == len(contexts) == len(negatives)

        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __len__(self):
        return len(self.centers)

    def __getitem__(self,index):
        return (self.centers[index],self.contexts[index],self.negatives[index])

def collate_batch(batch):
    # 因为前面的max_window_size是随机数所以，这里我们需要把向量对齐
    max_len = max(len(ct) + len(ng) for cr,ct,ng in batch)
    centers,conntexts_negatives,masks,labels = [],[],[],[]
    for cr,ct,ng in batch:
        centers += [cr]
        current_len = len(ct) + len(ng)
        conntexts_negatives += [ct + ng + [0] * (max_len - current_len)]
        # mask 是一个掩码向量，用来计算loss时候排出补进去的空缺值的影响
        masks += [[1] * current_len + [0] * (max_len - current_len)]
        labels += [[1] * len(ct) + [0] * (max_len - len(ct))]
    return torch.tensor(centers).view(-1,1),torch.tensor(conntexts_negatives),torch.tensor(masks),torch.tensor(labels)


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

class embedding(nn.Module):
    def __init__(self,id2word,embed_size: int = 100):
        super().__init__()
        self.embed_size = embed_size
        self.id2word = id2word
        

        self.embed_v = nn.Embedding(num_embeddings = len(self.id2word),embedding_dim = self.embed_size)
        self.embed_u = nn.Embedding(num_embeddings = len(self.id2word),embedding_dim = self.embed_size)

    def forward(self,centers,contexts_negatives):
        v = self.embed_v(centers)
        u = self.embed_u(contexts_negatives)
        pred = torch.bmm(v,u.permute(0,2,1))
        return pred

class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self): # none mean sum
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
    def forward(self, inputs, targets, mask=None):
        """
        input – Tensor shape: (batch_size, len)
        target – Tensor of the same shape as input
        """
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
        return res.mean(dim=1)

def model_fn(model,device,batch,criterion):

    center,context_negative,mask,label = batch
    # 这么写不行的!!!变量移不过去
    """for cc in [center,context_negative,mask,label]:
        cc = cc.to(device)"""
    model = model.to(device)
    center = center.to(device)
    context_negative = context_negative.to(device)
    mask = mask.to(device)
    label = label.to(device)
    predict = model(center,context_negative)
    # 平均到每一个非填充值的误差均值
    loss = (criterion(predict.view(label.shape),label,mask) * mask.shape[1] / mask.sum(dim = 1)).mean()

    return loss
def parse_args():
    config = {
        "batch_size":512,
        "n_workers":8,
        "lr":1e-3,
        "total_epoch":100,
        "n": None
    }
    return config
def main(batch_size,n_workers,lr,total_epoch,n):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] now use {device}")
    wvcy = WVCY(n)
    idx2word = wvcy.idx2word
    word2idx = wvcy.word2idx
    print(f"length of word : {len(idx2word)}")
    embed_size :int = 100
    model = embedding(id2word = idx2word,embed_size = embed_size).to(device)
    

    all_centers,all_contexts,all_negatives = wvcy.centers_contexts_nagatives
    print('[Initialize] finish initialize data')
    dataset = MyDataset(all_centers,all_contexts,all_negatives)
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = n_workers,
        collate_fn = collate_batch,
        shuffle = True
    )
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    criterion = SigmoidBinaryCrossEntropyLoss()
    pbar = tqdm(total = total_epoch,ncols = 0,desc = "train",unit = "epoch")
    for epoch in range(total_epoch):
        loss_collector = []
        for batch in tqdm(dataloader):
            model.train()
            loss = model_fn(model,device,batch,criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_collector.append(loss.cpu().item())
        epoch_avg_loss = sum(loss_collector) / len(loss_collector)
        pbar.update()
        pbar.set_postfix(
            loss = f"{epoch_avg_loss:.4f}",epoch = epoch
        )
    pbar.close()
    return model.embed_v.weight.data,word2idx




if __name__ == "__main__":
    wv,word2idx = main(**parse_args())
    wv = wv.cpu().numpy()
    joblib.dump(wv,"/root/autodl-tmp/PAPER/DATA/lexicon/wordvec/wordvec.pkl")
    joblib.dump(word2idx,"/root/autodl-tmp/PAPER/DATA/lexicon/wordvec/word2idx.pkl")
    os.system("shutdown")
    



