from collections import OrderedDict
import random
import pyhash
import numpy as np
from statistics import median
import matplotlib.pyplot as plt
from prettytable import PrettyTable

random.seed(42)
f=open('Q4_train.data','r')
l=[x.strip().split() for x in f.readlines()]
stream=[]
for x in l:
    stream.append((int(x[1]),int(x[2])))
actual={}
for p in stream:
    if p[0] in actual:
        actual[p[0]]+=p[1]
    else:
        actual[p[0]]=p[1]
max1000=OrderedDict(sorted(actual.items(), key=lambda t: t[1]))
max1000=list(max1000.items())
max1000=max1000[-1000:]
query_list=random.sample(max1000,100)
def cm_sketch(k):
    global stream
    w=5
    d=k//w
    hash_functions=[pyhash.metro_64(seed=i) for i in range(w)]
    table=np.zeros((w,d))
    for p in stream:
        for i in range(w):
            j=hash_functions[i](str(p[0]))%d
            table[i][j]+=p[1]
    return [table,hash_functions]

def cs_sketch(k):
    global stream
    w=5
    d=k//w
    hash_functions=[pyhash.metro_64(seed = i) for i in range(w)]
    g_hash_functions=[pyhash.metro_64(seed = i) for i in range(w,2*w)]
    table=np.zeros((w,d))
    for p in stream:
        for i in range(w):
            j=hash_functions[i](str(p[0]))%d
            sign=2*(g_hash_functions[i](str(p[0]))%2)-1
            table[i][j]+=sign*p[1]
    return [table,hash_functions,g_hash_functions]

    
def misra_gries_sketch(k):
    global stream
    freq={}
    for p in stream:
        if p[0] in freq:
            freq[p[0]]+=p[1]
        elif len(freq)<k-1:
            freq[p[0]]=p[1]
        else:
            n=[]
            for i in freq:
                freq[i]-=p[1]
                if freq[i]<=0:
                    n.append(freq[i])
            if len(n)>0:
                m=-min(n)
                freq[p[0]]=m
                d=[]
                for i in freq:
                    freq[i]+=m
                    if freq[i]==0:
                        d.append(i)
                for i in d:
                    del freq[i]
    return freq

def cm_query(table,hash_functions,x,d):
    w=5
    ans=[]
    for i in range(w):
        j=hash_functions[i](str(x))%d
        ans.append(table[i][j])
    return min(ans)

def cs_query(table,hash_functions,g_hash_functions,x,d):
    w=5
    ans=[]
    for i in range(w):
        j=hash_functions[i](str(x))%d
        sign=2*(g_hash_functions[i](str(x))%2)-1
        ans.append(sign*table[i][j])
    return median(ans)

def mg_query(freq,x):
    if x in freq:
        return freq[x]
    return 0

cm=[]
cs=[]
mg=[]
k_values=[100,200,500,1000,2000]
for k in k_values:
    cm.append(cm_sketch(k))
    cs.append(cs_sketch(k))
    mg.append(misra_gries_sketch(k))



cm_errlist=[]
cs_errlist=[]
mg_errlist=[]
for i in range(len(k_values)):
    err_cm=0
    err_cs=0
    err_mg=0
    for j in query_list:
        x=j[0]
        y=j[1]
        err_cm+=abs(cm_query(cm[i][0],cm[i][1],x,cm[i][0].shape[1])-y)/y
        err_cs+=abs(cs_query(cs[i][0],cs[i][1],cs[i][2],x,cs[i][0].shape[1])-y)/y
        err_mg+=abs(mg_query(mg[i],x)-y)/y
    cm_errlist.append(err_cm/100)
    cs_errlist.append(err_cs/100)
    mg_errlist.append(err_mg/100)

t=PrettyTable(['Sketch\\k']+k_values)
t.add_row(['CM Sketch']+cm_errlist)
t.add_row(['CS Sketch']+cs_errlist)
t.add_row(['Misra-Gries Sketch']+mg_errlist)
print(t)

plt.figure(1)
plt.plot(k_values,cm_errlist,label="CM Sketch")
plt.plot(k_values,cs_errlist,label="CS Sketch")
plt.plot(k_values,mg_errlist,label="Misra-Gries Sketch")
plt.xlabel("k")
plt.ylabel("Error")
plt.legend()
plt.title('Error v/s k for CM, CS and Misra-Gries Sketch')
plt.show()
plt.close()
