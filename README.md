# Combining transfer learning and active learning for text classification
A small student project for the Advanced topics in Machine Learning course at DTU.

As I work in the fields of natural language processing (NLP) and social science, the most appropriate way for me to handle the course project is to apply the methods to the type of data I work with (text, often from online debates).


## Data files

**IBM Debater Argument Search Engine Dataset** from:

*Towards an argumentative content search engine using weak supervision*.Ran Levy, Ben Bogin, Shai Gretz, Ranit Aharonov and Noam Slonim. COLING 2018.

*Unsupervised corpus--wide claim detection*. Ran Levy, Shai Gretz, Benjamin Sznajder, Shay Hummel, Ranit Aharonov and Noam Slonim. 4th Workshop on Argument Mining, EMNLP, 2017.

I have augmented the gold labeled test set by splitting the 2500 samples into train, dev, and test sets. In the tsv files are IDs, labels, the sentences and the topic, respectively. Find the original data at https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml to get the content, which I have excluded, described in readme_test_set.txt (i.e. main concept, query pattern, DNN score and url).


**big.txt** from https://www.kaggle.com/bittlingmayer/spelling / http://norvig.com/spell-correct.html


## Notes
**AL.py** contains all the code needed for pool-based active learning with uncertainty sampling, QBC and a random baseline, using a Linear SVM and either transformer-based (BERT) or Bag of Words (for comparison) text representations.

**DP.py** contains code for calculating the "minimum edit distance" between two strings using dynamical programming. This was the beginning of an idea to make a spelling corrector (just needs a splash of Bayesian inference to be finished), and to perhaps consider an reinforcement learning approach similar to this: https://arxiv.org/pdf/1801.10467.pdf. I instead chose to continue with an active learning project, but the code still serves as a good example of how DP can be useful in NLP. Reinforcement learning, however, is not very convenient for most NLP problems, but NLP can be useful in some limited reinforcement learning problems such as text-based games.
