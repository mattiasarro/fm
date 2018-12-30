# Stance detection and domain adaptation

## Literature review

The solutions proposed by the [original paper] have surprisingly poor performance, usually not even outperforming the SVM baseline. There has been some work since then that has advanced the SotA for this task. [Dey et al. 2017](http://sentic.net/sentire2017dey.pdf) propose a solution with a very simple algorithm but heavy feature engineering process which has the highest reported F1 scores >5% than previous SotA, on both task A and B of the challenge. Approaches based on deep learning tend to underperform even the SVM baseline reported by the authors of the challenge, though some attention-based deep learning models perform at a similar level, or marginally better, at 0.5% higher F1 score [Zhou et al 2017](https://warwick.ac.uk/fac/sci/dcs/people/research/csrnaj/WISE2017.pdf). A more sophisticated approach uses a dynamic memory-augmented network (DMAN) to capture stance-indicative information for multiple related targets, and does multi-task prediction from this dynamic memory [Wei et al 2018](https://dl.acm.org/citation.cfm?id=3210145).

The good performance of a model that relies heavily on feature engineering could be explained by the small dataset size for this problem. Approaches using memory-augmented networks could prove useful for a wide range of NLP tasks including stance detection, but are outside the scope of this work. We focus our attention on attention-based deep learning models, which are competitive with the SVM baseline yet likely to continue improving in performance as the dataset size increases. Neural networks are also appealing due to their amenability to multi-task and transfer learning, which is quite relevant for our purposes.

## Choice of model

Absent in literature are approaches that use pre-trained NLP models that could be fine-tuned for a specific purpose, such as ULMFit and BERT, and contextual word embeddings such as ELMo. BERT is a promising approach since its attention mechanism is bidirectional, and it has most recently improved on SotA results in many NLP tasks. While this might not be too much of use in short tweets, it is likely a better encoder for lengthy news articles than e.g. bidirectional LSTMs.

## Vocabulary mismatch

The obvious problem with Twitter data is the usage of special / abbreviated vocabulary, hashtags, username handles, and incorrect spelling in general. This will cause a lot of out of vocabulary (OOV) words when using pretrained models, and will not generalise to a new domain, since it would not contain the same set of hashtags or users. It would be a good idea to replace abbreviations, slang and misspellings in tweets with a more standard vocabulary that is used across other domains (news, books, Wikipedia). This would lead to a bigger overlap between the vocabularies of the source domain (Tweets), target domain (news articles), and pretraining domain (Wikipedia). The contents of hastags (e.g. character-by-character) could also be useful for predicting stance, however it is not clear how might be transferred to a new domain. Using the raw text could be valid in some cases but misleading in others (e.g. a tweet critical of Hillary Clinton could still be tagged with #HillaryForPresident).

## Experiments

## Hyperparameter Tuning.

Grid search over two parameters was performed (with values suggested by BERT's authors: learning_rate in [5e-5, 3e-5, 2e-5], batch_size in [16, 32]) for 20 epochs per set of hyperparameters. The model was evaluated after every epoch (91 steps) on the test set provided by the SemEval competition. Note that in a real-world scenario, the final test set would be used at the very end and hyperparameters would be tuned via k-fold cross validation on the original train set.

The figure below shows the macro average F1 score of a subset of the experiments. The highest achieved score was 73.9% (learning rate = 2e-05, batch size = 16), however the model with the most desirable learning curve (slower convergence, stable and high overall F1) converged at around 72%, with the highest score 72.9% (learning rate = 3e-05, batch size = 16). All models converged between the steps 750 and 1250 (respectively 8 and 13.5 epochs).

![tuning](./tuning1.png)

## Hashtags.

Removing the hashtag degrades performance on the SemEval test set, but might be worth doing when the only purpose is to do domain adaptation.

# Transfer Learning Procedure

Alternate between:

* P1: predict sentiment & stance from tweet & topic
* P2: predict URL domain from title/description & topic

## Limitations of current models

While current naive models can enjoy some success, I believe human-level stance detection is not feasible without either (1) a massive labelled dataset that contains the many ways in which a stance can be  expressed towards a given topic or (2) more sophisticated models that don't simply predict the stance of a text towards a specific target, but instead extract the different topics and entities within a sentence and predicts different stances towards these. The work of (Wei et al 2018) seems like a step in the right direction.
