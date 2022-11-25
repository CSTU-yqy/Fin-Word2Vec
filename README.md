# Fin-Word2Vec
## Adding tokens of coporations to word2vec to capture the more valuable synonyms of words in financial scenario

We would never know how many styles of expressions would describe the anomalies of firms’ situations, so there shouldn’t be an unchanging lexicon that can work all the time, as well as the existent ones. Still, we could always find out listed firms that were in trouble. Drawing on an outcome-oriented approach to financial fraud identification, we could collect news from the year before the point when the company “fell in trouble”. According to the official documents disclosed by the CSRC on the change of the company's listing status, we define “listed firm is in trouble” refers to a listed company becoming “ST”.

We cut the text from step1 into words; then, we would select the words that can represent any situation that could be called “anomalies”; we call these seed words. (We use “seed” to refer to words which can help grow or generate more synonymous terms.) We combine the seed words with all the selected negative words from the existent sentiment lexicon into a bad seed words lexicon.

The Word2Vec model (Mikolov et al. 2013) is one of the most effective ways to expand words, which can effectively capture words with the same background words  as synonyms (for example, Jiang et al. 2019). However, we require the model to think outside the box for negative word augmentation. Using different expressions or reporting angles, the ideal model should capture as many words as possible in bad news about the same company. Based on this requirement, we try to add the temporal distribution of bad news to the traditional Word2Vec model so that it can capture words with similar background words while maintaining the ability to capture diverse expressions and reporting angles.

The “bad news hoarding” has been demonstrated by researchers who focus on different markets worldwide (for example, Skinner 1994; Kothari, Shu, and Wysocki 2009; Wang, Han, and Huang 2020), which provides potential evidence of the aggregation of bad news appearance. By looking at the distribution of a sample of firm-year bad news items, we found a significantly higher concentration of bad news than the rest of the news in China. The left panel clearly shows that the duration of a firm-year's bad news is more left-skewed than the firm-year's rest of the news, which means that we can frame the majority of a company's bad news within a short time window; At the same time, the right panel also shows the cumulative distribution of the duration of bad news, with more than 80% of firm-year's bad news lasting less than six months, which supports our assumption that bad news tends to be more concentrated. 

![picture 1](./figures/Picture_1.png)
Based on these findings, we improved the original Word2Vec model by adding a “Stock code & Period token ” to it, combining the characteristics of the distribution of bad news appearance so that it can better detect. I name this modified Word2vec model as Fin-Word2Vec(see Appendix). This change could help us to capture negative words in bad news about the same company but using completely different angles and expressions.

![picture 2](./figures/Picture_2.png)
