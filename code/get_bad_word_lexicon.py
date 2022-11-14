
if __name__ == "__main__":

    #合成总词典
    #bw1 = open("/root/autodl-tmp/PAPER/DATA/lexicon/bad_news_dictionary/raw_data/bad_words.txt","r").read().split("\n")
    bw2 = open("/root/autodl-tmp/PAPER/DATA/lexicon/bad_news_dictionary/use_fin_word2vec/bad_news_dictionary_selected.txt","r").read().split("\n")
    bw3 = open("/root/autodl-tmp/PAPER/DATA/lexicon/bad_news_dictionary/use_fin_word2vec/bad_word_augmentation.txt","r").read().split("\n")
    bw3 = [i.split(" ")[0] for i in bw3]
    word_set = (set(bw2) | set(bw3)) - set(['',"\t","\n"])
    with open("/root/autodl-tmp/PAPER/DATA/lexicon/bad_news_dictionary/use_fin_word2vec/bad_word_lexicon.txt","w") as f:

        for i in word_set:
            f.write(i + "\n")
        f.close()