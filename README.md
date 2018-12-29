# Stance detection and domain adaptation

## Literature review

The solutions proposed by the [original paper] have surprisingly poor performance, usually not even outperforming the SVM baseline. There has been some work since then that has advanced the SotA for this task. [http://sentic.net/sentire2017dey.pdf](Dey et al. 2017) propose a solution with a very simple algorithm but heavy feature engineering process which has the highest reported F1 scores >5% than previous SotA, on both task A and B of the challenge. Approaches based on deep learning tend to underperform even the SVM baseline reported by the authors of the challenge, though some attention-based deep learning models perform at a similar level, or marginally better, at 0.5% higher F1 score [https://warwick.ac.uk/fac/sci/dcs/people/research/csrnaj/WISE2017.pdf](Zhou et al 2017). A more sophisticated approach uses a dynamic memory-augmented network (DMAN) to capture stance-indicative information for multiple related targets, and does multi-task prediction from this dynamic memory.

The good performance of a model that relies heavily on feature engineering could be explained by the small dataset size for this problem. Approaches using memory-augmented networks could prove useful for a wide range of NLP tasks including stance detection, but are outside the scope of this work. We focus our attention on attention-based deep learning models, which are competitive with the SVM baseline yet likely to continue improving in performance as the dataset size increases. Neural networks are also appealing due to their amenability to multi-task and transfer learning, which is quite relevant for our purposes.

## Choice of model

Absent in literature are approaches that use pre-trained NLP models that could be fine-tuned for a specific purpose, such as ULMFit and BERT, and contextual word embeddings such as ELMo. BERT is a promising approach since its attention mechanism is bidirectional, and it has most recently improved on SotA results in many NLP tasks. While this might not be too much of use in short tweets, it is likely a better encoder for lengthy news articles than e.g. bidirectional LSTMs.

second phrase is our target

## Vocabulary mismatch

The obvious problem with Twitter data is the usage of special / abbreviated vocabulary, hashtags, username handles, and incorrect spelling in general. This will cause a lot of out of vocabulary (OOV) words when using pretrained models, and will not generalise to a new domain, since it would not contain the same set of hashtags or users. It would be a good idea to replace abbreviations, slang and misspellings in tweets with a more standard vocabulary that is used across other domains (news, books, Wikipedia). This would lead to a bigger overlap between the vocabularies of the source domain (Tweets), target domain (news articles), and pretraining domain (Wikipedia). The contents of hastags (e.g. character-by-character) could also be useful for predicting stance, however it is not clear how might be transferred to a new domain. Using the raw text could be valid in some cases but misleading in others (e.g. a tweet critical of Hilary Clinton could still be tagged with #HilaryForPresident).

## Transfer Learning Procedure

Alternate between:

* P1: predict sentiment & stance from tweet & topic
* P2: predict URL domain from title/description & topic

--
https://github.com/zhouyiwei/tsd/blob/master/model.py
