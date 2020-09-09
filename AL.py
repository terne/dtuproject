import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import sklearn.linear_model as lin
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import matplotlib.pyplot as plt
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
def transform(sentences):
    output = []
    for sent in sentences:
        inputs = tokenizer(sent, padding=True, truncation=True, return_tensors="pt")
        input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]
        outputs = model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0] # "Sequence of hidden-states at the output of the last layer of the model."
        pooler_output = outputs[1] # "Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining."
        # either output mean embedding vectors or the pooler output
        #output.append(pooler_output[0].detach().numpy().tolist())
        output.append(torch.mean(last_hidden_state[0],0).detach().numpy().tolist())
    return output

print("transforming Xpool...")
Xpool = transform(Xpool)
print(len(Xpool[0]))
#Xpool = Xpool.detach().numpy()
print("transforming Xtest...")
Xtest = transform(Xtest)
#Xtest = Xtest.detach().numpy()
print("done with transformation")

print("undersampling now..")
rus = RandomUnderSampler(random_state=42)
Xpool, ypool = rus.fit_sample(Xpool, ypool)
print('Resampled dataset shape {}'.format(Counter(ypool)))
Xpool, Xtest = np.array(Xpool), np.array(Xtest)
print("length of Xpool", len(Xpool))

Xpool_class0idx = [indel for indel,i in enumerate(ypool) if i==0]
Xpool_class1idx = [indel for indel,i in enumerate(ypool) if i==1]
#print(Xpool_class1idx)
addn=10 #samples to add each time
#randomize order of pool to avoid sampling the same subject sequentially
order=np.random.permutation(range(len(Xpool)))
order0 = np.random.permutation(Xpool_class0idx)
order1 = np.random.permutation(Xpool_class1idx)

ninit = 5 #initial samples
#initial training set
#trainset=order[:ninit]
trainset=np.random.permutation(np.append(order0[:ninit],order1[:ninit])) # 5 from each class
print("initial data point indices:",trainset)

Xtrain=np.take(Xpool,trainset,axis=0)
ytrain=np.take(ypool,trainset,axis=0)
#remove data from pool
poolidx=np.arange(len(Xpool),dtype=np.int)
poolidx=np.setdiff1d(poolidx,trainset)
print("length of poolidx", len(poolidx))

num_iterations=95

# classification models
#LR = lin.LogisticRegression(penalty='l2',C=1.)
clf = SVC(kernel="linear")


print("Beginning random sampling")
testacc=[]
for i in range(num_iterations):
    #Fit model
    clf.fit(Xtrain,ytrain)
    #predict on test set
    ye=clf.predict(Xtest)
    #calculate and accuracy and add to list
    accuracy = accuracy_score(ytest,ye)
    testacc.append((len(Xtrain),accuracy))
    random_indices = np.random.choice(poolidx,addn)
    Xtrain = np.concatenate((Xtrain,Xpool[random_indices]))
    ytrain = np.concatenate((ytrain,ypool[random_indices]))
    poolidx=np.setdiff1d(poolidx,random_indices)
    print('Model: Linear SVM, {} random samples, Acc: {}, samples left in pool: {}'.format(len(Xtrain),accuracy,len(poolidx)))

# Uncertainty sampling following the FMRI exercise notebook
#reset training set and pool but starting with the same 10 samples as before.
clf = None
clf = SVC(kernel="linear")
print("initial data point indices:",trainset)
Xtrain=np.take(Xpool,trainset,axis=0)
ytrain=np.take(ypool,trainset,axis=0)
poolidx=np.arange(len(Xpool),dtype=np.int)
poolidx=np.setdiff1d(poolidx,trainset)
testacc_uncertainty = []
print("Beginning AL iterations")
for i in range(num_iterations):
    # fit model, (want Linear SVM later)
    clf.fit(Xtrain,ytrain)
    pred = clf.predict(Xtest)
    accuracy = accuracy_score(ytest,pred)
    testacc_uncertainty.append(accuracy)
    #get label probabilities on unlabelled pool, LR:
    #ypool_p = clf.predict_proba(Xpool[poolidx])
    #select least confident max likely label - then sort in negative order - note the minus, LR:
    #ypool_p_sort_idx = np.argsort(-ypool_p.max(1))
    # get samples closest to the class seperating hyperplane, linear SVM:
    ypool_p = clf.decision_function(Xpool[poolidx])
    ypool_p_sort_idx = np.argsort(-np.abs(np.ravel(ypool_p)))

    #add to training set
    Xtrain=np.concatenate((Xtrain,Xpool[poolidx[ypool_p_sort_idx[-addn:]]]))
    ytrain=np.concatenate((ytrain,ypool[poolidx[ypool_p_sort_idx[-addn:]]]))
    #remove from pool
    poolidx=np.setdiff1d(poolidx,ypool_p_sort_idx[-addn:])
    print('Model: Linear SVM, {} samples (uncertainty sampling), Acc: {}, samples left in pool: {}'.format(len(Xtrain), accuracy, len(poolidx)))
print(pred)

# Query by commitee
clf = None
clf = SVC(kernel="linear")
testacc_qbc=[]
ncomm=10
#reset training set and pool but starting with the same 10 samples as before.
print("initial data point indices:",trainset)
Xtrain=np.take(Xpool,trainset,axis=0)
ytrain=np.take(ypool,trainset,axis=0)
poolidx=np.arange(len(Xpool),dtype=np.int)
poolidx=np.setdiff1d(poolidx,trainset)
print("Beginning QBC")
for i in range(num_iterations):
    ypool_lab = []
    for j in range(ncomm):
        #bootstrapping
        Xtr,ytr=resample(Xtrain,ytrain,stratify=ytrain)
        #fit
        clf.fit(Xtr, ytr)
        #predict
        ypool_lab.append(clf.predict(Xpool[poolidx]))
    #get probability of label for each class based on voting in the committee
    ypool_p=(np.mean(np.array(ypool_lab)==0,0),np.mean(np.array(ypool_lab)==1,0))
    ypool_p=np.array(ypool_p).T
    #Refit model in all training data
    clf.fit(Xtrain,ytrain)
    ye=clf.predict(Xtest)
    accuracy = accuracy_score(ytest,ye)
    testacc_qbc.append((len(Xtrain),accuracy))
    #select sample with maximum disagreement (least confident)
    ypool_p_sort_idx = np.argsort(-ypool_p.max(1)) #least confident
    #add to training set
    Xtrain=np.concatenate((Xtrain,Xpool[poolidx[ypool_p_sort_idx[-addn:]]]))
    ytrain=np.concatenate((ytrain,ypool[poolidx[ypool_p_sort_idx[-addn:]]]))
    #remove from pool
    print(len(ypool_p_sort_idx[-addn:]))
    poolidx=np.setdiff1d(poolidx,ypool_p_sort_idx[-addn:])
    print('Model: Linear SVM, {} samples (QBC), Acc: {}, samples left in pool: {}'.format(len(Xtrain), accuracy, len(poolidx)))

#Plot learning curve
plt.plot(*tuple(np.array(testacc).T))
plt.plot(*tuple(np.array(testacc_uncertainty).T))
plt.plot(*tuple(np.array(testacc_qbc).T))
#plt.plot(*tuple(np.array(testacc_emc).T));
#plt.legend(('random sampling','uncertainty sampling','QBC','EMC'));
plt.legend(('random sampling','uncertainty sampling','QBC'))
plt.savefig("learning_curves.png", dpi=100)
