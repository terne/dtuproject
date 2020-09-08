import numpy as np
import re
import time
from collections import Counter

source = "thankss"
target = "thanks"


def del_cost(char):
    return 1

def insert_cost(char):
    return 1

def sub_cost(source_char, target_char):
    if source_char==target_char:
        return 0
    else:
        return 2

def min_edit_distance(source, target):
    n= len(source)
    m= len(target)
    D = np.zeros(shape=(n+1,m+1))

    for i in range(1,n+1):
        D[i,0] = D[i-1,0]+1 # deletion cost of 1
    for j in range(1,m+1):
        D[0,j] = D[0,j-1]+1 # insertion cost of 1

    for i in range(1,n+1):
        for j in range(1,m+1):
            D[i,j] = min([D[i-1,j]+del_cost(source[i-1]),
                          D[i-1,j-1]+sub_cost(source[i-1],target[j-1]),
                          D[i,j-1]+insert_cost(target[j-1])])
    return D[n,m]


D = min_edit_distance(source, target)
print(D)

def words(text):
    return re.findall(r'\w+', text.lower())

with open("big.txt") as text:
    words = re.findall(r'\w+', text.read().lower())

word_counts = Counter(words)
vocab_size = len(word_counts)


# do bayesian inference
