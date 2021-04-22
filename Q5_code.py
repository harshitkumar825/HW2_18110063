import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data=pd.read_csv('ratings.csv').drop(columns=['timestamp'])
data=data.rename(columns={'userId':0,'movieId':1,'rating':2})
data=data.iloc[:10000]
data=data.sample(frac=1).reset_index(drop=True)
testlen=2000
trainlen=8000

train=data.iloc[:trainlen]
test=data.iloc[trainlen:].reset_index(drop=True)
users=max(data[0])
movies=max(data[1])
train_matrix=np.zeros(((users,movies)))


for i in range(trainlen):
    train_matrix[train.iloc[i,0]-1][train.iloc[i,1]-1]=train.iloc[i,2]


def k_rank_approx(k,matrix):    
    u,s,v_t = np.linalg.svd(matrix,full_matrices=False)    
    if k>=len(s):
        return matrix
    else:
        u_n=u[:,:k]
        s_n=np.diag(s[:k])
        v_n=v_t[:k,:]
        c=np.matmul(u_n,np.matmul(s_n,v_n))
    return c

error=[]
for k in range(1,101):
    print(k)
    approx=k_rank_approx(k,train_matrix)
    e=0
    for i in range(testlen):
        e+=(test.iloc[i,2]-approx[test.iloc[i,0]-1][test.iloc[i,1]-1])**2
    error.append(e)

user={}
movie={}
for i in range(trainlen):
    if train.iloc[i,0] in user:
        stat=user[train.iloc[i,0]]
        user[train.iloc[i,0]]=[stat[0]+train.iloc[i,2],stat[1]+1]
    else:
        user[train.iloc[i,0]]=[train.iloc[i,2],1]
    if train.iloc[i,1] in movie:
        stat=movie[train.iloc[i,1]]
        movie[train.iloc[i,1]]=[stat[0]+train.iloc[i,2],stat[1]+1]
    else:
        movie[train.iloc[i,1]]=[train.iloc[i,2],1]


for i in user:
    stat=user[i]
    user[i]=stat[0]/stat[1]
for i in movie:
    stat=movie[i]
    movie[i]=stat[0]/stat[1]
baseline=[]
for i in range(trainlen):
    baseline.append([user[train.iloc[i,0]],movie[train.iloc[i,1]],train.iloc[i,2]])
baseline= np.array(baseline)

testing=[]
actual=[]
for i in range(testlen):
    a=0
    b=0
    if test.iloc[i,0] in user:
        a=user[test.iloc[i,0]]
    if test.iloc[i,1] in movie:
        b=movie[test.iloc[i,1]]
    testing.append([a,b])
    actual.append(test.iloc[i,2])


testing=np.array(testing)
lr_model=LinearRegression(fit_intercept=False)
lr_model.fit(baseline[:,:2],baseline[:,2])
predicted=lr_model.predict(testing)
error_baseline=mean_squared_error(predicted,actual)*len(actual)


plt.figure(1)
plt.plot([i+1 for i in range(100)],[error_baseline for i in range(100)], color='blue',label='Baseline')
plt.plot([i+1 for i in range(100)],[error[i] for i in range(100)], color='orange',label='Low rank approx.')
plt.title('error vs k (Low Rank Approx) w/ Baseline error')
plt.xlabel('k')
plt.ylabel('error')
plt.legend()
plt.show()
plt.close()
