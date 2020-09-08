import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import sklearn.linear_model as lin
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
np.random.seed(42) # random seed to ensure same results but feel free to change

if torch.cuda.is_available():
    device = torch.device("cuda")
    #print(f'There are {torch.cuda.device_count()} GPU(s) available.')

data_path = "IBM_debater_argument_search_tabulated_testset/"
train = pd.read_csv(data_path+"IBMdebaterArgSearch-PremiseAndOneHypothesis_train.tsv", sep="\t", names=["ID","label", "sentence", "topic"])
test = pd.read_csv(data_path+"IBMdebaterArgSearch-PremiseAndOneHypothesis_test.tsv", sep="\t", names=["ID","label", "sentence", "topic"])
print(len(train[train.label==0])/len(train))
print(len(train[train.label==1])/len(train))
print(len(test[test.label==0])/len(test))
print(len(test[test.label==1])/len(test))
# the data is very unbalanced. need to handle that.
Xpool = [str(i).lower() for i in train.sentence.values]
ypool = train.label.values
Xtest = [str(i).lower() for i in test.sentence.values]
ytest = test.label.values
#print(train.head())
print(len(ypool), len(ytest))
print(Xpool[:10])


# use pretrained embeddings (transfer learning) – transformer-based (BERT).
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained('bert-base-uncased')
def transform(sents):
    output = []
    b_index = 0
    while b_index<len(sents)-5:
        for sentences in sents[:b_index+5]:
            inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
            input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]
            outputs = model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs[0] # "Sequence of hidden-states at the output of the last layer of the model."
            pooler_output = outputs[1] # "Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining."
            # either output mean embedding vectors or the pooler output
            output.append(pooler_output)
            b_index+=5
    return output

print("transforming Xpool...")
Xpool = transform(Xpool)
Xpool = Xpool.detach().numpy()
print("transforming Xtest...")
Xtest = transform(Xtest)
Xtest = Xtest.detach().numpy()

rus = RandomUnderSampler(random_state=42)
Xpool, ypool = rus.fit_sample(Xpool, ypool)
print('Resampled dataset shape {}'.format(Counter(ypool)))


Xpool_class0idx = [indel for indel,i in enumerate(ypool) if i==0]
Xpool_class1idx = [indel for indel,i in enumerate(ypool) if i==1]
#print(Xpool_class1idx)
addn=5 #samples to add each time
#randomize order of pool to avoid sampling the same subject sequentially
#order=np.random.permutation(range(len(Xpool)))
order0 = np.random.permutation(Xpool_class0idx)
order1 = np.random.permutation(Xpool_class1idx)

#samples in the pool
poolidx=np.arange(len(Xpool),dtype=np.int)
ninit = 5 #initial samples
#initial training set
#trainset=order[:ninit]
trainset=np.random.permutation(np.append(order0[:ninit],order1[:ninit])) # 5 from each class
print(trainset)

Xtrain=np.take(Xpool,trainset,axis=0)
ytrain=np.take(ypool,trainset,axis=0)
#remove data from pool
poolidx=np.arange(len(Xpool),dtype=np.int)
poolidx=np.setdiff1d(poolidx,trainset)

num_iterations=25 # move up to 100
model = lin.LogisticRegression(penalty='l2',C=1.)

# Uncertainty sampling following the FMRI exercise notebook
acc = []
print("Beginning AL iterations")
for i in range(num_iterations):
    # fit model, (want Linear SVM later)
    model.fit(Xtrain,ytrain)
    pred = model.predict(Xtest)
    accuracy = accuracy_score(ytest,pred)
    acc.append(accuracy)
    #get label probabilities on unlabelled pool
    ypool_p = model.predict_proba(Xpool[poolidx])
    #select least confident max likely label - then sort in negative order - note the minus
    ypool_p_sort_idx = np.argsort(-ypool_p.max(1))
    #add to training set
    Xtrain=np.concatenate((Xtrain,Xpool[poolidx[ypool_p_sort_idx[-addn:]]]))
    ytrain=np.concatenate((ytrain,ypool[poolidx[ypool_p_sort_idx[-addn:]]]))
    #remove from pool
    poolidx=np.setdiff1d(poolidx,ypool_p_sort_idx[-addn:])
    print('Model: LR, {} samples (uncertainty sampling), Acc: {}'.format(len(Xtrain), accuracy))


# Query by commitee
