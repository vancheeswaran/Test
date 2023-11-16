#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import gzip
import math
from collections import defaultdict
import numpy
from sklearn import linear_model
import random
import statistics


# In[2]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[3]:


answers = {}


# In[4]:


z = gzip.open("train.json.gz")


# In[5]:


dataset = []
for l in z:
    d = eval(l)
    dataset.append(d)


# In[6]:


z.close()


# In[7]:


### Question 1


# In[8]:


def MSE(y, ypred):
    ''' 
    Function that computes the Mean Square Error
    '''
    return (1/len(y))*numpy.sum(numpy.square(numpy.array(y)-numpy.array(ypred)))


# In[9]:


def MAE(y, ypred):
    '''
    Function that computes the Mean Absolute Error
    '''
    return (1/len(y))*numpy.sum(numpy.absolute(numpy.array(y)-numpy.array(ypred)))


# In[10]:


reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    u,i = d['userID'],d['gameID']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)
    
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['date'])
    
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['date'])


# In[11]:


def feat1(d):
    '''
    Feature vector generator for q1
    '''
    return [1,d['hours']]


# In[12]:


X = [feat1(d) for d in dataset]
y = [len(d['text']) for d in dataset]


# In[13]:


mod = linear_model.LinearRegression()
mod.fit(X,y)
predictions = mod.predict(X)


# In[14]:


theta_1=mod.coef_[1]
mse_q1 = MSE(y,predictions)


# In[15]:


answers['Q1'] = [theta_1, mse_q1]


# In[16]:


assertFloatList(answers['Q1'], 2)


# In[17]:


### Question 2


# In[18]:


median= statistics.median([d['hours'] for d in dataset]) # Computing the median
def feat2(d):
    '''
    Feature vector generator for q2
    '''
    var= d['hours']
    return [1,var, math.log2(var+1),math.sqrt(var), var > median ]


# In[19]:


X = [feat2(d) for d in dataset]


# In[20]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)


# In[21]:


mse_q2= MSE(y,predictions)


# In[22]:


answers['Q2'] = mse_q2


# In[23]:


assertFloat(answers['Q2'])


# In[24]:


### Question 3


# In[25]:


def feat3(d):
    '''
    Feature vector generator for q3
    '''
    var= d['hours']
    return [1,var>1,var>5,var>10, var>100,var > 1000 ]


# In[26]:


X = [feat3(d) for d in dataset]


# In[27]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)


# In[28]:


mse_q3= MSE(y,predictions)


# In[29]:


answers['Q3'] = mse_q3


# In[30]:


assertFloat(answers['Q3'])


# In[31]:


### Question 4


# In[32]:


def feat4(d):
    '''
    Feature vector generator for q4
    '''
    return [1, len(d['text'])]


# In[33]:


X = [feat4(d) for d in dataset]
y = [d['hours'] for d in dataset]


# In[34]:


print(statistics.mean(y))
print(statistics.median(y))
print(max(y))
print(min(y))


# In[35]:


import seaborn as sns
sns.displot(y, binwidth=500)


# In[36]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)


# In[37]:


mse = MSE(y,predictions)
mae = MAE(y,predictions)


# In[38]:


answers['Q4'] = [mse, mae, "We see that the data has a lot of outliers and therefore, MSE might not be the best choice to compare the performance. MAE is more robust to outliers and hence might seem to be a better option here. However, MAE might not penalise the outliers enough, MAE treats all errors the same way, therefore MSE potentially could be a better metric here."]


# In[39]:


assertFloatList(answers['Q4'][:2], 2)


# In[40]:


### Question 5


# In[41]:


y_trans = [d['hours_transformed'] for d in dataset]


# In[42]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)


# In[43]:


mse_trans =  MSE(y_trans,predictions_trans) # MSE using the transformed variable


# In[44]:


predictions_untrans = 2**predictions_trans-1 # Undoing the transformation


# In[45]:


mse_untrans = MSE(y,predictions_untrans)


# In[46]:


answers['Q5'] = [mse_trans, mse_untrans]


# In[47]:


assertFloatList(answers['Q5'], 2)


# In[48]:


### Question 6


# In[49]:


def feat6(d):
    '''
    Feature vector generator for q6
    '''
    vec=[0]*100
    for i in range(100):
        if(i==int(d['hours'])):
            vec[i]=1
        else:
            vec[99]=1
    return [1]+vec


# In[50]:


X = [feat6(d) for d in dataset]
y = [len(d['text']) for d in dataset]


# In[51]:


Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[52]:


models = []
mses_valid = []
mses_test=[]
bestC = 1000

for c in [1, 10, 100, 1000, 10000]:
    
    mod = linear_model.Ridge(c)
    mod.fit(Xtrain,ytrain)
    predictions_valid = mod.predict(Xvalid)
    predictions_test = mod.predict(Xtest)
    models.append(mod)
    mses_valid.append(MSE(yvalid,predictions_valid))
    mses_test.append(MSE(ytest,predictions_test))
    


# In[53]:


mse_valid = mses_valid[3]


# In[54]:


mse_test = mses_test[3]


# In[55]:


answers['Q6'] = [bestC, mse_valid, mse_test]


# In[56]:


assertFloatList(answers['Q6'], 3)


# In[57]:


### Question 7


# In[58]:


times = [d['hours_transformed'] for d in dataset]
median = statistics.median(times)


# In[59]:


nNotPlayed =  [d['hours']<1 for d in dataset].count(True)


# In[60]:


answers['Q7'] = [median, nNotPlayed]


# In[61]:


assertFloatList(answers['Q7'], 2)


# In[62]:


### Question 8


# In[63]:


def feat8(d):
    '''
    Feature vector generator for q8
    '''
    return [1, len(d['text'])]


# In[64]:


X = [feat8(d) for d in dataset]
y = [d['hours_transformed'] > median for d in dataset]


# In[65]:


mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(X,y)
predictions = mod.predict(X) # Binary vector of predictions


# In[66]:


def rates(predictions, y):
    '''
    Function that computes TP, FP, TN, FN and BER
    '''
    
    TP = numpy.sum(numpy.logical_and(predictions, y))
    FP = numpy.sum(numpy.logical_and(predictions, numpy.logical_not(y)))
    TN = numpy.sum(numpy.logical_and(numpy.logical_not(predictions), numpy.logical_not(y)))
    FN = numpy.sum(numpy.logical_and(numpy.logical_not(predictions), y))
    BER= 1 - 0.5*(TP / (TP + FN) + TN / (TN + FP))
    
    return TP, TN, FP, FN, BER


# In[67]:


TP, TN, FP, FN, BER = rates(predictions, y)


# In[68]:


answers['Q8'] = [TP, TN, FP, FN, BER]


# In[69]:


assertFloatList(answers['Q8'], 5)


# In[70]:


### Question 9


# In[71]:


scores = mod.decision_function(X)
scoreslabels = list(zip(scores, y))
scoreslabels.sort(reverse=True)
sortedlabels = [x[1] for x in scoreslabels]


# In[72]:


precs = []

for i in [5, 10, 100, 1000]:
    threshold= scoreslabels[i-1][0]
    val=0
    n=0
    for j in scoreslabels:
        if(threshold <= j[0]):
            print(j[0])
            val+=j[1]
            n+=1
        else:
            break
    precs.append(val / n)


# In[73]:


answers['Q9'] = precs


# In[74]:


assertFloatList(answers['Q9'], 4)


# In[75]:


### Question 10


# In[76]:


y_trans = [d['hours_transformed'] for d in dataset]


# In[77]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)


# In[78]:


predictions_thresh = 3.8 # Using a fixed threshold to make predictions
predictions= predictions_trans>=predictions_thresh


# In[79]:


TP, TN, FP, FN, BER = rates(predictions,y_trans)


# In[80]:


answers['Q10'] = [predictions_thresh, BER]


# In[81]:


assertFloatList(answers['Q10'], 2)


# In[82]:


### Question 11


# In[83]:


dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]


# In[84]:


userMedian = defaultdict(list)
itemMedian = defaultdict(list)

# Compute medians on training data

for u in reviewsPerUser:
    userMedian[u] = statistics.median([d['hours_transformed'] for d in reviewsPerUser[u]])
    
for i in reviewsPerItem:
    itemMedian[i] = statistics.median([d['hours_transformed'] for d in reviewsPerItem[i]])
    


# In[85]:


answers['Q11'] = [itemMedian['g35322304'], userMedian['u55351001']]


# In[86]:


assertFloatList(answers['Q11'], 2)


# In[87]:


### Question 12


# In[88]:


def f12(u,i):
    # Function returns a single value (0 or 1)
    if(i not in itemMedian):
        if(userMedian[u]>median):
            return 1
        return 0
    elif(itemMedian[i]>median):
        return 1
    else:
        return 0


# In[89]:


preds = [f12(d['userID'], d['gameID']) for d in dataTest]


# In[90]:


y = [d['hours_transformed']>median for d in dataTest]


# In[91]:


accuracy = [ True if(i==j) else False for i,j in zip(y,preds)].count(True)/len(y)


# In[92]:


answers['Q12'] = accuracy


# In[93]:


assertFloat(answers['Q12'])


# In[94]:


### Question 13


# In[95]:


usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}

for d in dataset:
    user,item = d['userID'], d['gameID']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)


# In[96]:


def Jaccard(s1, s2):
    '''
    Function that computes the Jaccard Similarity
    '''
    
    n = len(s1.intersection(s2))
    d = len(s1.union(s2))
    
    if(d==0):
        return 0

    return n/d


# In[97]:


def mostSimilar(i, func, N):
    '''
    Function that return the N most similar entities
    '''
    
    similarities = []
    users = usersPerItem[i]
    
    for item in usersPerItem:
        if item != i: 
            sim = func(users, usersPerItem[item])
            similarities.append((sim,item))
    
    return sorted(similarities, reverse=True)[:N]
    


# In[98]:


ms = mostSimilar(dataset[0]['gameID'], Jaccard, 10)


# In[99]:


answers['Q13'] = [ms[0][0], ms[-1][0]]


# In[100]:


assertFloatList(answers['Q13'], 2)


# In[101]:


### Question 14


# In[102]:


def mostSimilar14(i, func, N):
    
    '''
    Function that return the N most similar items
    '''
    
    similarities = []
    
    for item in usersPerItem:
        if item != i: 
            sim = func(i, item)
            similarities.append((sim,item))
    
    return sorted(similarities, reverse=True)[:N]
    


# In[103]:


ratingDict = {}

for d in dataset:
    u,i = d['userID'], d['gameID']
    lab = [1 if(d['hours_transformed']>median) else -1][0] # Set the label based on a rule
    ratingDict[(u,i)] = lab


# In[104]:


def Cosine(i1, i2):
    # Between two items
    
    intersection = usersPerItem[i1].intersection(usersPerItem[i2])
    n = 0
    d1 = 0
    d2 = 0
    for u in intersection:
        n += ratingDict[(u,i1)]*ratingDict[(u,i2)]
    for u in usersPerItem[i1]:
        d1 += ratingDict[(u,i1)]**2
    for u in usersPerItem[i2]:
        d2 += ratingDict[(u,i2)]**2
    d = math.sqrt(d1) * math.sqrt(d2)
    if d == 0: return 0
    return n / d


# In[105]:


ms = mostSimilar14(dataset[0]['gameID'], Cosine, 10)


# In[106]:


answers['Q14'] = [ms[0][0], ms[-1][0]]


# In[107]:


assertFloatList(answers['Q14'], 2)


# In[108]:


### Question 15


# In[109]:


ratingDict = {}

for d in dataset:
    u,i = d['userID'], d['gameID']
    lab = d['hours_transformed']# Set the label based on a rule
    ratingDict[(u,i)] = lab


# In[110]:


ms = mostSimilar14(dataset[0]['gameID'], Cosine, 10)


# In[111]:


answers['Q15'] = [ms[0][0], ms[-1][0]]


# In[112]:


assertFloatList(answers['Q15'], 2)


# In[113]:


f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




