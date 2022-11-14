from re import match
import joblib
import numpy as np
import pandas as pd
import torch
from bidict import bidict
from tqdm import tqdm

if __name__ == "__main__":
    state = "_org"
    word_vec = joblib.load("/root/autodl-tmp/PAPER/DATA/lexicon/wordvec/wordvec%s.pkl"%state)
    word2idx = bidict(joblib.load("/root/autodl-tmp/PAPER/DATA/lexicon/wordvec/word2idx%s.pkl"%state))
    bad_word_seed = list(set(open("/root/autodl-tmp/PAPER/DATA/lexicon/bad_news_dictionary/use_fin_word2vec/bad_news_dictionary_selected.txt").read().split("\n")) - set(['']))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word_vec = torch.tensor(word_vec)
    word_vec = word_vec.to(device)

    def get_similar_word(target_word,k,word_vec):
        global word2idx
        idx = word2idx[target_word]
        x = word_vec[idx]
        cos = torch.matmul(word_vec,x) / (torch.sum(word_vec * word_vec,dim = 1) * torch.sum(x * x) + 1e-9).sqrt()
        _,topk = torch.topk(cos,k = k + 1)
        topk = topk.cpu().numpy()
        return {
            target_word:[word2idx.inverse[j] for j in topk[1:]]
        }
    
    for k in [10]:
        for j in [10]:
            with open("/root/autodl-tmp/PAPER/DATA/lexicon/bad_news_dictionary/use_fin_word2vec/bad_word_that_are_bad_word_selected_synonyms.txt","w") as f:

                bad_word_list = []
                for word in tqdm(word2idx.keys()):
                    if word not in bad_word_seed:
                        tk = list(get_similar_word(word,k,word_vec).values())[0]
                        match_bad_word = set(tk) & set(bad_word_seed)
                        if match_bad_word:
                            for bw in match_bad_word:
                                match_bad_word = bw
                                match_bad_word_topk = list(get_similar_word(bw,j,word_vec).values())[0]
                                if word in match_bad_word_topk:
                                    rank = match_bad_word_topk.index(word)
                                    f.write( "\n" + "%s : %s ; %s"%(word,match_bad_word,rank))
                                    break
                f.close()
                        




    # this is hope to after iteration the word amount would converge at a acceptable amount, but we fail at last
    # k = 5
    # kk = 5
    # bad_words_num = 0
    # new_bad_words_num = len(bad_word_seed)
    # while new_bad_words_num > bad_words_num:
    #     bad_words_num = len(bad_word_seed)
    #     bad_word_dictionary = {}
    #     for bw in tqdm(bad_word_seed):
    #         if bw in word2idx.keys():
    #             bad_word_dictionary.update(get_similar_word(bw,k,word_vec))
    #     bad_word_list = []
    #     for _,v in bad_word_dictionary.items():
    #         v = [ii for ii in v if ii not in _ or _ in ii]
    #         bad_word_list += v[:kk]
    #     bad_word_seed = sorted(list(set(bad_word_list) | set(bad_word_seed)))
    #     new_bad_words_num = len(bad_word_seed)
    # bad_word_list = sorted(list(set(bad_word_seed)))
    # with open("/root/autodl-tmp/PAPER/DATA/lexicon/bad_news_dictionary/use_fin_word2vec/bad_words_dictionary_new.txt","w") as f:
    #     for b in bad_word_list:
    #         f.write( "\n" + b)
    #     f.close()
    





    


